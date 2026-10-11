use alvr_common::ViewParams;
use alvr_graphics::SceneDepth;
use alvr_packets::SceneDepthHeader;
use std::{collections::VecDeque, time::Duration};

const MAX_RECENT_FRAMES: usize = 64;
const MAX_PENDING_DEPTHS: usize = 8;
// Older depth is not used, for example if the app stopped submitting it
const MAX_DEPTH_AGE: Duration = Duration::from_millis(250);

// Matches the scene depth images (which arrive a few frames late) with the view params of the
// video frames they belong to, and selects the newest usable one.
#[derive(Default)]
pub struct SceneDepthBuffer {
    recent_view_params: VecDeque<(Duration, [ViewParams; 2])>,
    pending: VecDeque<(SceneDepthHeader, Vec<u8>)>,
    current: Option<(SceneDepthHeader, Vec<u8>, [ViewParams; 2])>,
}

impl SceneDepthBuffer {
    pub fn report_frame_view_params(&mut self, timestamp: Duration, view_params: [ViewParams; 2]) {
        self.recent_view_params.push_back((timestamp, view_params));
        if self.recent_view_params.len() > MAX_RECENT_FRAMES {
            self.recent_view_params.pop_front();
        }
    }

    pub fn push(&mut self, header: SceneDepthHeader, nearness: Vec<u8>) {
        self.pending.push_back((header, nearness));
        if self.pending.len() > MAX_PENDING_DEPTHS {
            self.pending.pop_front();
        }
    }

    pub fn get(&mut self, frame_timestamp: Duration) -> Option<SceneDepth<'_>> {
        let oldest_known = self.recent_view_params.front().map(|(t, _)| *t);

        // Resolve pending depths whose video frame is known; drop those that can never resolve
        let mut pending = VecDeque::new();
        for (header, nearness) in self.pending.drain(..) {
            if let Some((_, view_params)) = self
                .recent_view_params
                .iter()
                .find(|(t, _)| *t == header.timestamp)
            {
                if self
                    .current
                    .as_ref()
                    .is_none_or(|(current, ..)| header.timestamp > current.timestamp)
                {
                    self.current = Some((header, nearness, *view_params));
                }
            } else if oldest_known.is_none_or(|oldest| header.timestamp >= oldest) {
                pending.push_back((header, nearness));
            }
        }
        self.pending = pending;

        self.current
            .as_ref()
            .filter(|(header, ..)| header.timestamp + MAX_DEPTH_AGE >= frame_timestamp)
            .map(|(header, nearness, view_params)| SceneDepth {
                timestamp: header.timestamp,
                eye_resolution: header.eye_resolution,
                max_distance: header.max_distance,
                nearness,
                view_params: *view_params,
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alvr_common::glam::UVec2;

    fn header(ms: u64) -> SceneDepthHeader {
        SceneDepthHeader {
            timestamp: Duration::from_millis(ms),
            eye_resolution: UVec2::ONE,
            max_distance: 1.0,
        }
    }

    fn ms(ms: u64) -> Duration {
        Duration::from_millis(ms)
    }

    #[test]
    fn waits_for_view_params() {
        let mut buffer = SceneDepthBuffer::default();
        buffer.push(header(10), vec![1, 2]);
        assert!(buffer.get(ms(10)).is_none());

        buffer.report_frame_view_params(ms(10), [ViewParams::DUMMY; 2]);
        assert_eq!(buffer.get(ms(20)).unwrap().timestamp, ms(10));
    }

    #[test]
    fn keeps_newest_and_expires() {
        let mut buffer = SceneDepthBuffer::default();
        for t in [10, 20, 30] {
            buffer.report_frame_view_params(ms(t), [ViewParams::DUMMY; 2]);
        }
        buffer.push(header(30), vec![3, 3]);
        buffer.push(header(20), vec![2, 2]);
        assert_eq!(buffer.get(ms(40)).unwrap().timestamp, ms(30));

        assert!(buffer.get(ms(30) + MAX_DEPTH_AGE).is_some());
        assert!(buffer.get(ms(31) + MAX_DEPTH_AGE).is_none());
    }

    #[test]
    fn drops_unresolvable() {
        let mut buffer = SceneDepthBuffer::default();
        buffer.report_frame_view_params(ms(50), [ViewParams::DUMMY; 2]);
        buffer.push(header(10), vec![1, 1]);
        assert!(buffer.get(ms(50)).is_none());
        assert!(buffer.pending.is_empty());
    }
}
