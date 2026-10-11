#include "SceneDepth.h"
#include "alvr_server/Logger.h"
#include "alvr_server/Utils.h"
#include "alvr_server/bindings.h"

#include <d3dcompiler.h>
#include <string.h>

#pragma comment(lib, "d3dcompiler.lib")

using Microsoft::WRL::ComPtr;

namespace {

// Nearness 0 corresponds to this distance (meters, along the view axis). Beyond arm's reach, the
// scene can never occlude the hands.
const float MAX_DISTANCE = 1.275f;

// Linearizes the app depth with its projection matrix and keeps the nearest surface within the
// footprint of each output pixel. The output is the nearness: 1 at the eye, 0 at MAX_DISTANCE.
const char SHADER_SOURCE[] = R"(
cbuffer Params : register(b0) {
    float4 bounds;          // uMin, vMin, uMax, vMax
    float4 projection;      // m[2][2], m[2][3], m[3][2], m[3][3]
    float2 viewportOrigin;  // top left corner of this eye in the target, in pixels
    float2 eyeSize;         // size of an eye in the target, in pixels
    float maxDistance;
    float3 padding;
};

#ifdef MULTISAMPLED
Texture2DMS<float> depthTexture : register(t0);
#else
Texture2D<float> depthTexture : register(t0);
#endif

float4 vertex_main(uint id : SV_VertexID) : SV_Position {
    float2 uv = float2((id << 1) & 2, id & 2);
    return float4(uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
}

float LoadDepth(int2 texel) {
#ifdef MULTISAMPLED
    return depthTexture.Load(texel, 0);
#else
    return depthTexture.Load(int3(texel, 0));
#endif
}

float pixel_main(float4 position : SV_Position) : SV_Target {
    uint width, height;
#ifdef MULTISAMPLED
    uint samples;
    depthTexture.GetDimensions(width, height, samples);
#else
    uint levels;
    depthTexture.GetDimensions(0, width, height, levels);
#endif
    float2 textureSize = float2(width, height);

    float2 pixel = floor(position.xy - viewportOrigin);
    float2 uvStart = lerp(bounds.xy, bounds.zw, pixel / eyeSize);
    float2 uvEnd = lerp(bounds.xy, bounds.zw, (pixel + 1.0) / eyeSize);

    float nearest = maxDistance;
    [unroll] for (int y = 0; y < 4; y++) {
        [unroll] for (int x = 0; x < 4; x++) {
            float2 uv = lerp(uvStart, uvEnd, (float2(x, y) + 0.5) / 4.0);
            int2 texel = clamp(int2(uv * textureSize), int2(0, 0), int2(textureSize) - 1);
            float depth = LoadDepth(texel);

            // Invert the projection: clip.z = m22 * z + m23, clip.w = m32 * z + m33
            float z = (projection.y - depth * projection.w) / (depth * projection.z - projection.x);
            float distance = -z;
            if (distance > 0.0) {
                nearest = min(nearest, distance);
            }
        }
    }

    return saturate(1.0 - nearest / maxDistance);
}
)";

struct ShaderParams {
    float bounds[4];
    float projection[4];
    float viewportOrigin[2];
    float eyeSize[2];
    float maxDistance;
    float padding[3];
};
static_assert(sizeof(ShaderParams) % 16 == 0, "Constant buffer size must be a multiple of 16");

ComPtr<ID3DBlob> CompileShader(const char* entryPoint, const char* target, bool multisampled) {
    D3D_SHADER_MACRO multisampledDefines[] = { { "MULTISAMPLED", "1" }, { nullptr, nullptr } };

    ComPtr<ID3DBlob> code;
    ComPtr<ID3DBlob> errors;
    HRESULT hr = D3DCompile(
        SHADER_SOURCE,
        sizeof(SHADER_SOURCE) - 1,
        "SceneDepth",
        multisampled ? multisampledDefines : nullptr,
        nullptr,
        entryPoint,
        target,
        D3DCOMPILE_OPTIMIZATION_LEVEL3,
        0,
        &code,
        &errors
    );
    if (FAILED(hr)) {
        Error(
            "SceneDepth: failed to compile %s: %s",
            entryPoint,
            errors ? (const char*)errors->GetBufferPointer() : "unknown error"
        );
        return nullptr;
    }

    return code;
}

DXGI_FORMAT DepthViewFormat(DXGI_FORMAT textureFormat) {
    switch (textureFormat) {
    case DXGI_FORMAT_R32_TYPELESS:
        return DXGI_FORMAT_R32_FLOAT;
    case DXGI_FORMAT_R32G8X24_TYPELESS:
        return DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
    case DXGI_FORMAT_R24G8_TYPELESS:
        return DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
    case DXGI_FORMAT_R16_TYPELESS:
        return DXGI_FORMAT_R16_UNORM;
    default:
        return DXGI_FORMAT_UNKNOWN;
    }
}

// Saves and restores the pipeline state touched by SceneDepth, so that the other renderers sharing
// the immediate context are not affected.
class ContextStateGuard {
public:
    explicit ContextStateGuard(ID3D11DeviceContext* context)
        : m_context(context) {
        m_context->OMGetRenderTargets(
            D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, m_renderTargets, &m_depthStencilView
        );
        m_context->OMGetBlendState(&m_blendState, m_blendFactor, &m_sampleMask);
        m_context->OMGetDepthStencilState(&m_depthStencilState, &m_stencilRef);
        m_context->RSGetState(&m_rasterizerState);
        m_context->RSGetViewports(&m_viewportCount, nullptr);
        m_context->RSGetViewports(&m_viewportCount, m_viewports);
        m_context->IAGetInputLayout(&m_inputLayout);
        m_context->IAGetPrimitiveTopology(&m_topology);
        m_context->VSGetShader(&m_vertexShader, nullptr, nullptr);
        m_context->PSGetShader(&m_pixelShader, nullptr, nullptr);
        m_context->PSGetShaderResources(0, 1, &m_shaderResource);
        m_context->PSGetConstantBuffers(0, 1, &m_constantBuffer);
    }

    ~ContextStateGuard() {
        m_context->OMSetRenderTargets(
            D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, m_renderTargets, m_depthStencilView
        );
        m_context->OMSetBlendState(m_blendState, m_blendFactor, m_sampleMask);
        m_context->OMSetDepthStencilState(m_depthStencilState, m_stencilRef);
        m_context->RSSetState(m_rasterizerState);
        m_context->RSSetViewports(m_viewportCount, m_viewports);
        m_context->IASetInputLayout(m_inputLayout);
        m_context->IASetPrimitiveTopology(m_topology);
        m_context->VSSetShader(m_vertexShader, nullptr, 0);
        m_context->PSSetShader(m_pixelShader, nullptr, 0);
        m_context->PSSetShaderResources(0, 1, &m_shaderResource);
        m_context->PSSetConstantBuffers(0, 1, &m_constantBuffer);

        for (auto view : m_renderTargets) {
            if (view)
                view->Release();
        }
        IUnknown* objects[] = { m_depthStencilView, m_blendState,    m_depthStencilState,
                                m_rasterizerState,  m_inputLayout,   m_vertexShader,
                                m_pixelShader,      m_shaderResource, m_constantBuffer };
        for (auto object : objects) {
            if (object)
                object->Release();
        }
    }

private:
    ID3D11DeviceContext* m_context;
    ID3D11RenderTargetView* m_renderTargets[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT] = {};
    ID3D11DepthStencilView* m_depthStencilView = nullptr;
    ID3D11BlendState* m_blendState = nullptr;
    FLOAT m_blendFactor[4] = {};
    UINT m_sampleMask = 0;
    ID3D11DepthStencilState* m_depthStencilState = nullptr;
    UINT m_stencilRef = 0;
    ID3D11RasterizerState* m_rasterizerState = nullptr;
    UINT m_viewportCount = D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE;
    D3D11_VIEWPORT m_viewports[D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE] = {};
    ID3D11InputLayout* m_inputLayout = nullptr;
    D3D11_PRIMITIVE_TOPOLOGY m_topology = D3D11_PRIMITIVE_TOPOLOGY_UNDEFINED;
    ID3D11VertexShader* m_vertexShader = nullptr;
    ID3D11PixelShader* m_pixelShader = nullptr;
    ID3D11ShaderResourceView* m_shaderResource = nullptr;
    ID3D11Buffer* m_constantBuffer = nullptr;
};

} // namespace

SceneDepth::SceneDepth(std::shared_ptr<CD3DRender> d3dRender)
    : m_d3dRender(d3dRender) { }

bool SceneDepth::IsSupportedDepthFormat(DXGI_FORMAT format) {
    return DepthViewFormat(format) != DXGI_FORMAT_UNKNOWN;
}

bool SceneDepth::Initialize() {
    if (m_initialized) {
        return true;
    }
    if (m_initializationFailed) {
        return false;
    }
    m_initializationFailed = true;

    auto device = m_d3dRender->GetDevice();

    auto vertexCode = CompileShader("vertex_main", "vs_5_0", false);
    auto pixelCode = CompileShader("pixel_main", "ps_5_0", false);
    auto pixelCodeMultisampled = CompileShader("pixel_main", "ps_5_0", true);
    if (!vertexCode || !pixelCode || !pixelCodeMultisampled) {
        return false;
    }

    if (FAILED(device->CreateVertexShader(
            vertexCode->GetBufferPointer(), vertexCode->GetBufferSize(), nullptr, &m_vertexShader
        ))
        || FAILED(device->CreatePixelShader(
            pixelCode->GetBufferPointer(), pixelCode->GetBufferSize(), nullptr, &m_pixelShader
        ))
        || FAILED(device->CreatePixelShader(
            pixelCodeMultisampled->GetBufferPointer(),
            pixelCodeMultisampled->GetBufferSize(),
            nullptr,
            &m_pixelShaderMultisampled
        ))) {
        Error("SceneDepth: failed to create shaders");
        return false;
    }

    D3D11_BUFFER_DESC bufferDesc = {};
    bufferDesc.ByteWidth = sizeof(ShaderParams);
    bufferDesc.Usage = D3D11_USAGE_DYNAMIC;
    bufferDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    bufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    if (FAILED(device->CreateBuffer(&bufferDesc, nullptr, &m_constantBuffer))) {
        Error("SceneDepth: failed to create constant buffer");
        return false;
    }

    D3D11_TEXTURE2D_DESC targetDesc = {};
    targetDesc.Width = EYE_WIDTH * 2;
    targetDesc.Height = EYE_HEIGHT;
    targetDesc.MipLevels = 1;
    targetDesc.ArraySize = 1;
    targetDesc.Format = DXGI_FORMAT_R8_UNORM;
    targetDesc.SampleDesc.Count = 1;
    targetDesc.Usage = D3D11_USAGE_DEFAULT;
    targetDesc.BindFlags = D3D11_BIND_RENDER_TARGET;
    if (FAILED(device->CreateTexture2D(&targetDesc, nullptr, &m_target))
        || FAILED(device->CreateRenderTargetView(m_target.Get(), nullptr, &m_targetView))) {
        Error("SceneDepth: failed to create render target");
        return false;
    }

    D3D11_TEXTURE2D_DESC stagingDesc = targetDesc;
    stagingDesc.Usage = D3D11_USAGE_STAGING;
    stagingDesc.BindFlags = 0;
    stagingDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    for (auto& staging : m_staging) {
        if (FAILED(device->CreateTexture2D(&stagingDesc, nullptr, &staging.texture))) {
            Error("SceneDepth: failed to create staging texture");
            return false;
        }
    }

    m_readbackBuffer.resize(EYE_WIDTH * 2 * EYE_HEIGHT);

    Info("SceneDepth: initialized");
    m_initializationFailed = false;
    m_initialized = true;
    return true;
}

bool SceneDepth::RenderEye(
    int eye,
    ID3D11Texture2D* depthTexture,
    const vr::VRTextureBounds_t& bounds,
    const vr::HmdMatrix44_t& projection
) {
    auto device = m_d3dRender->GetDevice();
    auto context = m_d3dRender->GetContext();

    D3D11_TEXTURE2D_DESC desc;
    depthTexture->GetDesc(&desc);

    bool multisampled = desc.SampleDesc.Count > 1;

    D3D11_SHADER_RESOURCE_VIEW_DESC viewDesc = {};
    viewDesc.Format = DepthViewFormat(desc.Format);
    if (multisampled) {
        viewDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DMS;
    } else {
        viewDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        viewDesc.Texture2D.MostDetailedMip = 0;
        viewDesc.Texture2D.MipLevels = 1;
    }

    ComPtr<ID3D11ShaderResourceView> view;
    if (viewDesc.Format == DXGI_FORMAT_UNKNOWN || !(desc.BindFlags & D3D11_BIND_SHADER_RESOURCE)
        || FAILED(device->CreateShaderResourceView(depthTexture, &viewDesc, &view))) {
        return false;
    }

    ShaderParams params = {};
    params.bounds[0] = bounds.uMin;
    params.bounds[1] = bounds.vMin;
    params.bounds[2] = bounds.uMax;
    params.bounds[3] = bounds.vMax;
    params.projection[0] = projection.m[2][2];
    params.projection[1] = projection.m[2][3];
    params.projection[2] = projection.m[3][2];
    params.projection[3] = projection.m[3][3];
    params.viewportOrigin[0] = (float)(eye * EYE_WIDTH);
    params.viewportOrigin[1] = 0.0f;
    params.eyeSize[0] = (float)EYE_WIDTH;
    params.eyeSize[1] = (float)EYE_HEIGHT;
    params.maxDistance = MAX_DISTANCE;

    D3D11_MAPPED_SUBRESOURCE mapped;
    if (FAILED(context->Map(m_constantBuffer.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) {
        return false;
    }
    memcpy(mapped.pData, &params, sizeof(params));
    context->Unmap(m_constantBuffer.Get(), 0);

    D3D11_VIEWPORT viewport = {};
    viewport.TopLeftX = params.viewportOrigin[0];
    viewport.TopLeftY = 0.0f;
    viewport.Width = (float)EYE_WIDTH;
    viewport.Height = (float)EYE_HEIGHT;
    viewport.MinDepth = 0.0f;
    viewport.MaxDepth = 1.0f;

    context->RSSetViewports(1, &viewport);
    context->PSSetShader(
        multisampled ? m_pixelShaderMultisampled.Get() : m_pixelShader.Get(), nullptr, 0
    );
    context->PSSetShaderResources(0, 1, view.GetAddressOf());
    context->Draw(3, 0);

    // Unbind, the depth texture is used as a depth stencil by its owner
    ID3D11ShaderResourceView* nullView = nullptr;
    context->PSSetShaderResources(0, 1, &nullView);

    return true;
}

void SceneDepth::Process(
    ID3D11Texture2D* depthTextures[2],
    const vr::VRTextureBounds_t bounds[2],
    const vr::HmdMatrix44_t projections[2],
    uint64_t targetTimestampNs
) {
    if (!depthTextures[0] || !depthTextures[1] || !Initialize()) {
        return;
    }

    // Without a perspective projection the depth cannot be linearized
    if (projections[0].m[3][2] == 0.0f || projections[1].m[3][2] == 0.0f) {
        return;
    }

    auto context = m_d3dRender->GetContext();

    bool rendered;
    {
        ContextStateGuard stateGuard(context);

        float clearColor[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        context->ClearRenderTargetView(m_targetView.Get(), clearColor);
        context->OMSetRenderTargets(1, m_targetView.GetAddressOf(), nullptr);
        context->OMSetBlendState(nullptr, nullptr, 0xffffffff);
        context->OMSetDepthStencilState(nullptr, 0);
        context->RSSetState(nullptr);
        context->IASetInputLayout(nullptr);
        context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        context->VSSetShader(m_vertexShader.Get(), nullptr, 0);
        context->PSSetConstantBuffers(0, 1, m_constantBuffer.GetAddressOf());

        rendered = RenderEye(0, depthTextures[0], bounds[0], projections[0])
            && RenderEye(1, depthTextures[1], bounds[1], projections[1]);
    }

    if (!rendered) {
        return;
    }

    // If the readback of the oldest frame is still not done, it is dropped
    auto& staging = m_staging[m_nextStaging];
    m_nextStaging = (m_nextStaging + 1) % STAGING_COUNT;

    context->CopyResource(staging.texture.Get(), m_target.Get());
    staging.targetTimestampNs = targetTimestampNs;
    staging.pending = true;
}

void SceneDepth::Poll() {
    if (!m_initialized) {
        return;
    }

    auto context = m_d3dRender->GetContext();

    // Oldest first
    for (int i = 0; i < STAGING_COUNT; i++) {
        auto& staging = m_staging[(m_nextStaging + i) % STAGING_COUNT];
        if (!staging.pending) {
            continue;
        }

        D3D11_MAPPED_SUBRESOURCE mapped;
        HRESULT hr = context->Map(
            staging.texture.Get(), 0, D3D11_MAP_READ, D3D11_MAP_FLAG_DO_NOT_WAIT, &mapped
        );
        if (hr == DXGI_ERROR_WAS_STILL_DRAWING) {
            // Newer frames cannot be ready either
            break;
        }
        staging.pending = false;
        if (FAILED(hr)) {
            continue;
        }

        uint32_t rowSize = EYE_WIDTH * 2;
        for (uint32_t y = 0; y < EYE_HEIGHT; y++) {
            memcpy(
                &m_readbackBuffer[y * rowSize],
                (const uint8_t*)mapped.pData + y * mapped.RowPitch,
                rowSize
            );
        }
        context->Unmap(staging.texture.Get(), 0);

        SendSceneDepth(
            staging.targetTimestampNs, m_readbackBuffer.data(), EYE_WIDTH, EYE_HEIGHT, MAX_DISTANCE
        );
    }
}
