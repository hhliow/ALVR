// Scene depth of the streamed frame, used on the client to occlude the passthrough hands behind
// virtual objects. Only the depth range the hands can reach is kept, quantized to 8 bit "nearness":
// 255 at the eye, 0 at or beyond `max_distance`. Distant geometry therefore becomes long runs of
// zeros, which a trivial run-length encoding removes.

use alvr_common::{
    anyhow::{Result, bail},
    glam::UVec2,
};
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SceneDepthHeader {
    /// Same as the timestamp of the corresponding video frame
    pub timestamp: Duration,
    /// The image contains the left and right eye side by side, each of this size. Rows go from
    /// top to bottom.
    pub eye_resolution: UVec2,
    /// Distance in meters (along the view axis) corresponding to nearness 0
    pub max_distance: f32,
}

// Encoding: a zero byte is followed by a byte with the length of the run of zeros (1..=255); any
// other byte is a literal.
pub fn encode_scene_depth(nearness: &[u8]) -> Vec<u8> {
    let mut encoded = Vec::with_capacity(nearness.len() / 4);

    let mut i = 0;
    while i < nearness.len() {
        if nearness[i] == 0 {
            let run = nearness[i..]
                .iter()
                .take(u8::MAX as usize)
                .take_while(|&&v| v == 0)
                .count();
            encoded.extend([0, run as u8]);
            i += run;
        } else {
            encoded.push(nearness[i]);
            i += 1;
        }
    }

    encoded
}

pub fn decode_scene_depth(encoded: &[u8], expected_len: usize) -> Result<Vec<u8>> {
    let mut nearness = Vec::with_capacity(expected_len);

    let mut bytes = encoded.iter();
    while let Some(&value) = bytes.next() {
        if value == 0 {
            let Some(&run) = bytes.next() else {
                bail!("Truncated scene depth zero run");
            };
            if run == 0 {
                bail!("Invalid scene depth zero run");
            }
            nearness.resize(nearness.len() + run as usize, 0);
        } else {
            nearness.push(value);
        }

        if nearness.len() > expected_len {
            bail!("Scene depth data is larger than expected");
        }
    }

    if nearness.len() != expected_len {
        bail!(
            "Scene depth data size mismatch: expected {expected_len}, got {}",
            nearness.len()
        );
    }

    Ok(nearness)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let mut data = vec![0; 1000];
        data[10] = 1;
        data[11] = 255;
        data[600..620].fill(128);
        data[999] = 7;

        let encoded = encode_scene_depth(&data);
        assert!(encoded.len() < 40);
        assert_eq!(decode_scene_depth(&encoded, data.len()).unwrap(), data);
    }

    #[test]
    fn roundtrip_no_zeros() {
        let data = (1..=255).collect::<Vec<u8>>();
        let encoded = encode_scene_depth(&data);
        assert_eq!(encoded, data);
        assert_eq!(decode_scene_depth(&encoded, data.len()).unwrap(), data);
    }

    #[test]
    fn rejects_malformed() {
        assert!(decode_scene_depth(&[0], 1).is_err());
        assert!(decode_scene_depth(&[0, 0], 0).is_err());
        assert!(decode_scene_depth(&[0, 10], 5).is_err());
        assert!(decode_scene_depth(&[1, 2], 3).is_err());
        assert!(decode_scene_depth(&[], 0).unwrap().is_empty());
    }
}
