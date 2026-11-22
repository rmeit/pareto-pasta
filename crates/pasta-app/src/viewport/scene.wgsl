// Polyscope-style scene shaders for the pasta viewport.
//
// All colour math happens in linear space; the colour attachment is an sRGB view so the
// hardware encodes on write. Colour-map polynomials produce sRGB values and are linearised.

struct Uniforms {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    camera_pos: vec4<f32>,     // xyz eye position
    base_color: vec4<f32>,     // linear rgb
    back_color: vec4<f32>,     // linear rgb
    ghost_color: vec4<f32>,    // linear rgb, a = alpha
    plate_color: vec4<f32>,    // linear rgb, a = alpha
    line_color: vec4<f32>,     // linear rgb, a = alpha
    wire_color: vec4<f32>,     // linear rgb, a = strength
    slice_point: vec4<f32>,    // w = 1 when enabled
    slice_normal: vec4<f32>,
    scalar: vec4<f32>,         // min, max, has_scalars, colormap id
    flags: vec4<f32>,          // material id, wireframe, ground visible, dark
    ground_origin: vec4<f32>,  // xyz origin on plane, w = shadow half extent
    ground_u: vec4<f32>,       // xyz axis, w = ground quad half size
    ground_v: vec4<f32>,       // xyz axis, w = grid spacing
    up: vec4<f32>,             // xyz up, w = 1 if camera is above the ground
    viewport: vec4<f32>,       // width px, height px, line width px, unused
    bg_top: vec4<f32>,
    bg_bottom: vec4<f32>,
    ground_color: vec4<f32>,   // rgb, a = shadow strength
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var shadow_tex: texture_2d<f32>;
@group(0) @binding(2) var shadow_smp: sampler;

// ---------------------------------------------------------------- helpers

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let lo = c / 12.92;
    let hi = pow((max(c, vec3<f32>(0.0)) + 0.055) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, c <= vec3<f32>(0.04045));
}

fn viridis(t_in: f32) -> vec3<f32> {
    let t = clamp(t_in, 0.0, 1.0);
    let c0 = vec3<f32>(0.2777273272234177, 0.005407344544966578, 0.3340998053353061);
    let c1 = vec3<f32>(0.1050930431085774, 1.404613529898575, 1.384590162594685);
    let c2 = vec3<f32>(-0.3308618287255563, 0.214847559468213, 0.09509516302823659);
    let c3 = vec3<f32>(-4.634230498983486, -5.799100973351585, -19.33244095627987);
    let c4 = vec3<f32>(6.228269936347081, 14.17993336680509, 56.69055260068105);
    let c5 = vec3<f32>(4.776384997670288, -13.74514537774601, -65.35303263337234);
    let c6 = vec3<f32>(-5.435455855934631, 4.645852612178535, 26.3124352495832);
    return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
}

fn turbo(t_in: f32) -> vec3<f32> {
    let x = clamp(t_in, 0.0, 1.0);
    let r4 = vec4<f32>(0.13572138, 4.61539260, -42.66032258, 132.13108234);
    let g4 = vec4<f32>(0.09140261, 2.19418839, 4.84296658, -14.18503333);
    let b4 = vec4<f32>(0.10667330, 12.64194608, -60.58204836, 110.36276771);
    let r2 = vec2<f32>(-152.94239396, 59.28637943);
    let g2 = vec2<f32>(4.27729857, 2.82956604);
    let b2 = vec2<f32>(-89.90310912, 27.34824973);
    let v4 = vec4<f32>(1.0, x, x * x, x * x * x);
    let v2 = v4.zw * v4.z;
    return vec3<f32>(dot(v4, r4) + dot(v2, r2), dot(v4, g4) + dot(v2, g2), dot(v4, b4) + dot(v2, b2));
}

// Moreland's smooth cool-warm diverging map, piecewise-linear through 9 sRGB samples.
fn coolwarm(t_in: f32) -> vec3<f32> {
    var k = array<vec3<f32>, 9>(
        vec3<f32>(0.2298, 0.2987, 0.7537),
        vec3<f32>(0.3509, 0.4652, 0.8879),
        vec3<f32>(0.4846, 0.6236, 0.9739),
        vec3<f32>(0.6196, 0.7414, 0.9954),
        vec3<f32>(0.8654, 0.8654, 0.8654),
        vec3<f32>(0.9567, 0.7469, 0.6481),
        vec3<f32>(0.9578, 0.6033, 0.4828),
        vec3<f32>(0.8873, 0.4001, 0.3135),
        vec3<f32>(0.7057, 0.0156, 0.1502),
    );
    let s = clamp(t_in, 0.0, 1.0) * 8.0;
    let i = min(u32(floor(s)), 7u);
    let f = s - f32(i);
    return mix(k[i], k[i + 1u], f);
}

fn colormap(t: f32) -> vec3<f32> {
    let id = u.scalar.w;
    var c: vec3<f32>;
    if id < 0.5 {
        c = viridis(t);
    } else if id < 1.5 {
        c = coolwarm(t);
    } else {
        c = turbo(t);
    }
    return srgb_to_linear(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)));
}

// Analytic "matcap": shading depends only on the view-space normal.
// material: 0 = clay, 1 = wax, 2 = flat
fn matcap(albedo: vec3<f32>, n: vec3<f32>, material: f32) -> vec3<f32> {
    if material > 1.5 {
        return albedo;
    }
    let l = normalize(vec3<f32>(-0.32, 0.48, 0.82));   // key light, upper left, toward viewer
    let ndl = dot(n, l);
    let wrap = clamp((ndl + 0.15) / 1.15, 0.0, 1.0);    // slightly wrapped diffuse
    let facing = clamp(n.z, 0.0, 1.0);
    let ambient = 0.17 + 0.07 * n.y;                     // sky/ground hemisphere tint
    // matcap-style falloff toward grazing angles, then a narrow, gentle rim light
    let falloff = mix(0.6, 1.0, sqrt(facing));
    var col = albedo * (ambient + 0.9 * wrap * wrap) * falloff;
    let rim = pow(1.0 - facing, 5.0);
    col += vec3<f32>(0.10) * rim;
    if material > 0.5 {
        let h = normalize(l + vec3<f32>(0.0, 0.0, 1.0));
        let spec = pow(max(dot(n, h), 0.0), 36.0);
        col = col * 0.92 + vec3<f32>(0.42) * spec;
    }
    return col;
}

fn slice_discard(world: vec3<f32>) -> bool {
    return u.slice_point.w > 0.5 && dot(world - u.slice_point.xyz, u.slice_normal.xyz) > 0.0;
}

// ---------------------------------------------------------------- background

struct BgOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) t: f32,
};

@vertex
fn vs_background(@builtin(vertex_index) vi: u32) -> BgOut {
    let x = f32((vi << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(vi & 2u) * 2.0 - 1.0;
    var o: BgOut;
    o.pos = vec4<f32>(x, y, 1.0, 1.0);
    o.t = y * 0.5 + 0.5;
    return o;
}

@fragment
fn fs_background(in: BgOut) -> @location(0) vec4<f32> {
    let t = clamp(in.t, 0.0, 1.0);
    return vec4<f32>(mix(u.bg_bottom.rgb, u.bg_top.rgb, t * t * (3.0 - 2.0 * t)), 1.0);
}

// ---------------------------------------------------------------- ground

struct GroundOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) uv: vec2<f32>,
};

@vertex
fn vs_ground(@builtin(vertex_index) vi: u32) -> GroundOut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
    );
    let c = corners[vi];
    let half = u.ground_u.w;
    let uv = c * half;
    let world = u.ground_origin.xyz + u.ground_u.xyz * uv.x + u.ground_v.xyz * uv.y;
    var o: GroundOut;
    o.pos = u.view_proj * vec4<f32>(world, 1.0);
    o.world = world;
    o.uv = uv;
    return o;
}

fn grid_line(coord: vec2<f32>, spacing: f32) -> f32 {
    let g = coord / spacing;
    let w = fwidth(g);
    let d = abs(fract(g - 0.5) - 0.5) / max(w, vec2<f32>(1e-6));
    return 1.0 - min(min(d.x, d.y), 1.0);
}

@fragment
fn fs_ground(in: GroundOut) -> @location(0) vec4<f32> {
    if u.up.w < 0.5 {
        discard;
    }
    let half = u.ground_u.w;
    let spacing = u.ground_v.w;
    var col = u.ground_color.rgb;
    let minor = grid_line(in.uv, spacing);
    let major = grid_line(in.uv, spacing * 5.0);
    let dark = u.flags.w;
    let line_col = select(col * 0.80, col * 1.6 + vec3<f32>(0.02), dark > 0.5);
    col = mix(col, line_col, max(minor * 0.45, major * 0.9));

    // soft shadow from the blurred top-down coverage map
    let e = u.ground_origin.w;
    let suv = vec2<f32>(in.uv.x / e * 0.5 + 0.5, 0.5 - in.uv.y / e * 0.5);
    var shadow = 0.0;
    if all(suv > vec2<f32>(0.0)) && all(suv < vec2<f32>(1.0)) {
        shadow = textureSampleLevel(shadow_tex, shadow_smp, suv, 0.0).r;
    }
    col *= 1.0 - u.ground_color.a * shadow;

    // fade out toward the edge of the quad
    let r = length(in.uv) / half;
    let alpha = 1.0 - smoothstep(0.45, 1.0, r);
    return vec4<f32>(col, alpha);
}

// ---------------------------------------------------------------- meshes

struct MeshIn {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) scalar: f32,
};

struct MeshOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) vnormal: vec3<f32>,
    @location(2) scalar: f32,
    @location(3) bary: vec3<f32>,
};

@vertex
fn vs_mesh(in: MeshIn, @builtin(vertex_index) vi: u32) -> MeshOut {
    var o: MeshOut;
    o.pos = u.view_proj * vec4<f32>(in.pos, 1.0);
    o.world = in.pos;
    o.vnormal = (u.view * vec4<f32>(in.normal, 0.0)).xyz;
    o.scalar = in.scalar;
    let k = vi % 3u;
    o.bary = vec3<f32>(f32(k == 0u), f32(k == 1u), f32(k == 2u));
    return o;
}

fn shade_normal(vnormal: vec3<f32>, front: bool) -> vec3<f32> {
    var n = vnormal;
    let len = length(n);
    if len < 1e-12 {
        n = vec3<f32>(0.0, 0.0, 1.0);
    } else {
        n = n / len;
    }
    if !front {
        n = -n;
    }
    return n;
}

fn wire_mix(col: vec3<f32>, bary: vec3<f32>, d: vec3<f32>) -> vec3<f32> {
    if u.flags.y < 0.5 {
        return col;
    }
    let a = smoothstep(vec3<f32>(0.0), d * 0.9, bary);
    let edge = 1.0 - min(min(a.x, a.y), a.z);
    return mix(col, u.wire_color.rgb, edge * u.wire_color.a);
}

@fragment
fn fs_mesh(in: MeshOut, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    let dbary = fwidth(in.bary);
    if slice_discard(in.world) {
        discard;
    }
    let n = shade_normal(in.vnormal, front);
    var albedo: vec3<f32>;
    if u.scalar.z > 0.5 {
        let range = u.scalar.y - u.scalar.x;
        let t = (in.scalar - u.scalar.x) / select(range, 1.0, abs(range) < 1e-30);
        albedo = colormap(t);
        if !front {
            albedo *= 0.72;
        }
    } else {
        albedo = select(u.back_color.rgb, u.base_color.rgb, front);
    }
    var col = matcap(albedo, n, u.flags.x);
    col = wire_mix(col, in.bary, dbary);
    return vec4<f32>(col, 1.0);
}

@fragment
fn fs_ghost(in: MeshOut, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    if slice_discard(in.world) {
        discard;
    }
    let n = shade_normal(in.vnormal, front);
    let col = matcap(u.ghost_color.rgb, n, 0.0);
    return vec4<f32>(col, u.ghost_color.a);
}

@fragment
fn fs_plate(in: MeshOut, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    let n = shade_normal(in.vnormal, front);
    let col = matcap(u.plate_color.rgb, n, 1.0);
    return vec4<f32>(col, u.plate_color.a);
}

// ---------------------------------------------------------------- shadow coverage (top-down)

struct ShadowOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) height: f32,
};

@vertex
fn vs_shadow(in: MeshIn) -> ShadowOut {
    let p = in.pos - u.ground_origin.xyz;
    let e = u.ground_origin.w;
    var o: ShadowOut;
    o.pos = vec4<f32>(dot(p, u.ground_u.xyz) / e, dot(p, u.ground_v.xyz) / e, 0.5, 1.0);
    o.world = in.pos;
    o.height = dot(p, u.up.xyz);
    return o;
}

@fragment
fn fs_shadow(in: ShadowOut) -> @location(0) vec4<f32> {
    if slice_discard(in.world) {
        discard;
    }
    // objects high above the ground cast lighter shadows
    let k = 1.0 - 0.6 * clamp(in.height / u.ground_origin.w, 0.0, 1.0);
    return vec4<f32>(k, k, k, k);
}

// ---------------------------------------------------------------- thick lines

struct LineIn {
    @location(0) a: vec3<f32>,
    @location(1) b: vec3<f32>,
    @location(2) ts: vec2<f32>,   // t in {0,1}, side in {-1,1}
};

struct LineOut {
    @builtin(position) pos: vec4<f32>,
};

@vertex
fn vs_line(in: LineIn) -> LineOut {
    let ca = u.view_proj * vec4<f32>(in.a, 1.0);
    let cb = u.view_proj * vec4<f32>(in.b, 1.0);
    let vp = u.viewport.xy;
    // guard against points behind the camera
    let wa = max(ca.w, 1e-4);
    let wb = max(cb.w, 1e-4);
    let sa = ca.xy / wa * vp * 0.5;
    let sb = cb.xy / wb * vp * 0.5;
    var dir = sb - sa;
    if length(dir) < 1e-6 {
        dir = vec2<f32>(1.0, 0.0);
    }
    dir = normalize(dir);
    let perp = vec2<f32>(-dir.y, dir.x);
    let c = select(ca, cb, in.ts.x > 0.5);
    let off = perp * in.ts.y * u.viewport.z * 0.5 / (vp * 0.5);
    var o: LineOut;
    o.pos = vec4<f32>(c.xy + off * c.w, c.z - 2e-5 * c.w, c.w);
    return o;
}

@fragment
fn fs_line() -> @location(0) vec4<f32> {
    return u.line_color;
}
