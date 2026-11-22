// Separable Gaussian blur for the ground shadow coverage map.

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_smp: sampler;

struct Out {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_full(@builtin(vertex_index) vi: u32) -> Out {
    let x = f32((vi << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(vi & 2u) * 2.0 - 1.0;
    var o: Out;
    o.pos = vec4<f32>(x, y, 0.0, 1.0);
    o.uv = vec2<f32>(x * 0.5 + 0.5, 0.5 - y * 0.5);
    return o;
}

fn blur(uv: vec2<f32>, dir: vec2<f32>) -> vec4<f32> {
    let texel = 1.0 / vec2<f32>(textureDimensions(src_tex));
    let sigma = 6.0;
    var acc = 0.0;
    var wsum = 0.0;
    for (var i = -15; i <= 15; i++) {
        let x = f32(i);
        let w = exp(-0.5 * x * x / (sigma * sigma));
        acc += w * textureSampleLevel(src_tex, src_smp, uv + dir * texel * x, 0.0).r;
        wsum += w;
    }
    let v = acc / wsum;
    return vec4<f32>(v, v, v, v);
}

@fragment
fn fs_blur_h(in: Out) -> @location(0) vec4<f32> {
    return blur(in.uv, vec2<f32>(1.0, 0.0));
}

@fragment
fn fs_blur_v(in: Out) -> @location(0) vec4<f32> {
    return blur(in.uv, vec2<f32>(0.0, 1.0));
}
