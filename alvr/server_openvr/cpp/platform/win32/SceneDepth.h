#pragma once

#include <memory>
#include <stdint.h>
#include <vector>

#include <d3d11.h>
#include <wrl.h>

#include "alvr_server/openvr_driver_wrap.h"
#include "shared/d3drender.h"

// Extracts the depth buffer submitted by the SteamVR app, keeps only the range the hands can reach
// and sends it to the client at low resolution, where it is used to occlude the passthrough hands
// behind virtual objects.
//
// Each eye is downsampled on the GPU into an 8 bit "nearness" image (255 at the eye, 0 at or
// beyond MAX_DISTANCE), taking the nearest surface of each output pixel's footprint. The result is
// read back asynchronously a few frames later, so that the present thread never waits for the GPU.
class SceneDepth {
public:
    static const uint32_t EYE_WIDTH = 128;
    static const uint32_t EYE_HEIGHT = 128;

    explicit SceneDepth(std::shared_ptr<CD3DRender> d3dRender);

    // Returns true if this texture format can be bound as a depth stencil and sampled.
    static bool IsSupportedDepthFormat(DXGI_FORMAT format);

    // Must be called with exclusive access to the immediate context. depthTextures can be the
    // same texture for both eyes, with different bounds.
    void Process(
        ID3D11Texture2D* depthTextures[2],
        const vr::VRTextureBounds_t bounds[2],
        const vr::HmdMatrix44_t projections[2],
        uint64_t targetTimestampNs
    );

    // Sends the frames whose readback is complete. Must be called with exclusive access to the
    // immediate context.
    void Poll();

private:
    static const int STAGING_COUNT = 3;

    struct Staging {
        Microsoft::WRL::ComPtr<ID3D11Texture2D> texture;
        uint64_t targetTimestampNs = 0;
        bool pending = false;
    };

    bool Initialize();
    bool RenderEye(
        int eye,
        ID3D11Texture2D* depthTexture,
        const vr::VRTextureBounds_t& bounds,
        const vr::HmdMatrix44_t& projection
    );

    std::shared_ptr<CD3DRender> m_d3dRender;
    bool m_initialized = false;
    bool m_initializationFailed = false;

    Microsoft::WRL::ComPtr<ID3D11VertexShader> m_vertexShader;
    Microsoft::WRL::ComPtr<ID3D11PixelShader> m_pixelShader;
    Microsoft::WRL::ComPtr<ID3D11PixelShader> m_pixelShaderMultisampled;
    Microsoft::WRL::ComPtr<ID3D11Buffer> m_constantBuffer;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> m_target;
    Microsoft::WRL::ComPtr<ID3D11RenderTargetView> m_targetView;

    Staging m_staging[STAGING_COUNT];
    int m_nextStaging = 0;
    std::vector<uint8_t> m_readbackBuffer;
};
