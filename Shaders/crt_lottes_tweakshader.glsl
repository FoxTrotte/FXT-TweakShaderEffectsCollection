/*
 * CRT Lottes port for Tweak Shader / After Effects.
 *
 * Original public-domain shader by Timothy Lottes:
 * https://github.com/libretro/glsl-shaders/blob/master/crt/shaders/crt-lottes.glsl
 *
 * Adaptation notes:
 * - Converted from libretro's combined vertex/fragment format to Tweak Shader.
 * - Uses one After Effects image input.
 * - Adds emulated_pixel_size so HD footage can be treated as a lower-resolution
 *   signal before scanline reconstruction.
 * - Adds bloom and alpha switches while retaining the original defaults.
 */

#pragma utility_block(ShaderInputs)
layout(push_constant) uniform ShaderInputs {
    float time;
    float time_delta;
    float frame_rate;
    uint  frame_index;
    vec4  mouse;
    vec4  date;
    vec3  resolution;
    uint  pass_index;
};

layout(location=0) out vec4 out_color;

#pragma sampler(name="linear_sampler", linear, clamp)
layout(set=0, binding=1) uniform sampler linear_sampler;

#pragma input(image, name="input_image")
layout(set=0, binding=2) uniform texture2D input_image;

/* Original CRT Lottes parameters. */
#pragma input(float, name=hard_scan, default=-8.0, min=-20.0, max=0.0)
#pragma input(float, name=hard_pixel, default=-3.0, min=-20.0, max=0.0)
#pragma input(float, name=warp_x, default=0.031, min=0.0, max=0.125)
#pragma input(float, name=warp_y, default=0.041, min=0.0, max=0.125)
#pragma input(float, name=mask_dark, default=0.5, min=0.0, max=2.0)
#pragma input(float, name=mask_light, default=1.5, min=0.0, max=2.0)
#pragma input(float, name=brightness_boost, default=1.0, min=0.0, max=2.0)
#pragma input(float, name=bloom_pixel_softness, default=-1.5, min=-2.0, max=-0.5)
#pragma input(float, name=bloom_scan_softness, default=-2.0, min=-4.0, max=-1.0)
#pragma input(float, name=bloom_amount, default=0.15, min=0.0, max=1.0)
#pragma input(float, name=filter_shape, default=2.0, min=0.0, max=10.0)

/* AE-oriented controls.
 * shadow_mask values:
 *   0 = Off
 *   1 = Compressed TV
 *   2 = Aperture grille
 *   3 = Stretched VGA (original default)
 *   4 = VGA
 */
#pragma input(int, name=shadow_mask, default=3, values=[0, 1, 2, 3, 4], labels=["Off", "Compressed TV", "Aperture Grille", "Stretched VGA", "VGA"])
#pragma input(float, name=emulated_pixel_size, default=2.0, min=1.0, max=12.0)
#pragma input(float, name=mask_scale, default=1.0, min=0.25, max=4.0)
#pragma input(bool, name=linearize_srgb, default=true)
#pragma input(bool, name=enable_bloom, default=true)
#pragma input(bool, name=preserve_alpha, default=true)

layout(set=0, binding=3) uniform CustomInputs {
    float hard_scan;
    float hard_pixel;
    float warp_x;
    float warp_y;
    float mask_dark;
    float mask_light;
    float brightness_boost;
    float bloom_pixel_softness;
    float bloom_scan_softness;
    float bloom_amount;
    float filter_shape;
    int   shadow_mask;
    float emulated_pixel_size;
    float mask_scale;
    int   linearize_srgb;
    int   enable_bloom;
    int   preserve_alpha;
};

vec2 output_size()
{
    return max(resolution.xy, vec2(1.0));
}

vec2 virtual_source_size()
{
    return max(floor(output_size() / max(emulated_pixel_size, 1.0)), vec2(1.0));
}

bool uv_outside(vec2 uv)
{
    return any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)));
}

float to_linear_channel(float value)
{
    if (linearize_srgb == 0) {
        return value;
    }

    value = max(value, 0.0);
    return value <= 0.04045
        ? value / 12.92
        : pow((value + 0.055) / 1.055, 2.4);
}

vec3 to_linear(vec3 color)
{
    return vec3(
        to_linear_channel(color.r),
        to_linear_channel(color.g),
        to_linear_channel(color.b));
}

float to_srgb_channel(float value)
{
    if (linearize_srgb == 0) {
        return value;
    }

    value = max(value, 0.0);
    return value < 0.0031308
        ? value * 12.92
        : 1.055 * pow(value, 1.0 / 2.4) - 0.055;
}

vec3 to_srgb(vec3 color)
{
    return vec3(
        to_srgb_channel(color.r),
        to_srgb_channel(color.g),
        to_srgb_channel(color.b));
}

vec4 sample_raw(vec2 uv)
{
    if (uv_outside(uv)) {
        return vec4(0.0);
    }
    return texture(sampler2D(input_image, linear_sampler), uv);
}

/* Nearest emulated sample at a floating source position and integer texel offset. */
vec3 fetch_emulated(vec2 position, vec2 offset)
{
    vec2 source_size = virtual_source_size();
    vec2 sample_uv = (floor(position * source_size + offset) + vec2(0.5)) / source_size;

    if (uv_outside(sample_uv)) {
        return vec3(0.0);
    }

    vec3 color = texture(sampler2D(input_image, linear_sampler), sample_uv).rgb;
    return to_linear(brightness_boost * color);
}

/* Distance in emulated pixels to the nearest source texel center. */
vec2 distance_to_texel(vec2 position)
{
    vec2 source_position = position * virtual_source_size();
    return -((source_position - floor(source_position)) - vec2(0.5));
}

/* Original generalized Gaussian kernel. */
float gaussian_weight(float position, float scale)
{
    return exp2(scale * pow(abs(position), max(filter_shape, 0.0001)));
}

vec3 horizontal_3(vec2 position, float line_offset)
{
    vec3 left   = fetch_emulated(position, vec2(-1.0, line_offset));
    vec3 center = fetch_emulated(position, vec2( 0.0, line_offset));
    vec3 right  = fetch_emulated(position, vec2( 1.0, line_offset));

    float distance_x = distance_to_texel(position).x;
    float weight_left   = gaussian_weight(distance_x - 1.0, hard_pixel);
    float weight_center = gaussian_weight(distance_x,       hard_pixel);
    float weight_right  = gaussian_weight(distance_x + 1.0, hard_pixel);
    float weight_sum = max(weight_left + weight_center + weight_right, 1e-8);

    return (left * weight_left + center * weight_center + right * weight_right) / weight_sum;
}

vec3 horizontal_5(vec2 position, float line_offset)
{
    vec3 sample_a = fetch_emulated(position, vec2(-2.0, line_offset));
    vec3 sample_b = fetch_emulated(position, vec2(-1.0, line_offset));
    vec3 sample_c = fetch_emulated(position, vec2( 0.0, line_offset));
    vec3 sample_d = fetch_emulated(position, vec2( 1.0, line_offset));
    vec3 sample_e = fetch_emulated(position, vec2( 2.0, line_offset));

    float distance_x = distance_to_texel(position).x;
    float weight_a = gaussian_weight(distance_x - 2.0, hard_pixel);
    float weight_b = gaussian_weight(distance_x - 1.0, hard_pixel);
    float weight_c = gaussian_weight(distance_x,       hard_pixel);
    float weight_d = gaussian_weight(distance_x + 1.0, hard_pixel);
    float weight_e = gaussian_weight(distance_x + 2.0, hard_pixel);
    float weight_sum = max(weight_a + weight_b + weight_c + weight_d + weight_e, 1e-8);

    return (sample_a * weight_a + sample_b * weight_b + sample_c * weight_c +
            sample_d * weight_d + sample_e * weight_e) / weight_sum;
}

vec3 horizontal_7_bloom(vec2 position, float line_offset)
{
    vec3 sample_a = fetch_emulated(position, vec2(-3.0, line_offset));
    vec3 sample_b = fetch_emulated(position, vec2(-2.0, line_offset));
    vec3 sample_c = fetch_emulated(position, vec2(-1.0, line_offset));
    vec3 sample_d = fetch_emulated(position, vec2( 0.0, line_offset));
    vec3 sample_e = fetch_emulated(position, vec2( 1.0, line_offset));
    vec3 sample_f = fetch_emulated(position, vec2( 2.0, line_offset));
    vec3 sample_g = fetch_emulated(position, vec2( 3.0, line_offset));

    float distance_x = distance_to_texel(position).x;
    float weight_a = gaussian_weight(distance_x - 3.0, bloom_pixel_softness);
    float weight_b = gaussian_weight(distance_x - 2.0, bloom_pixel_softness);
    float weight_c = gaussian_weight(distance_x - 1.0, bloom_pixel_softness);
    float weight_d = gaussian_weight(distance_x,       bloom_pixel_softness);
    float weight_e = gaussian_weight(distance_x + 1.0, bloom_pixel_softness);
    float weight_f = gaussian_weight(distance_x + 2.0, bloom_pixel_softness);
    float weight_g = gaussian_weight(distance_x + 3.0, bloom_pixel_softness);
    float weight_sum = max(weight_a + weight_b + weight_c + weight_d +
                           weight_e + weight_f + weight_g, 1e-8);

    return (sample_a * weight_a + sample_b * weight_b + sample_c * weight_c +
            sample_d * weight_d + sample_e * weight_e + sample_f * weight_f +
            sample_g * weight_g) / weight_sum;
}

/* Five-tap horizontal filter using the bloom kernel. */
vec3 horizontal_5_bloom(vec2 position, float line_offset)
{
    vec3 sample_a = fetch_emulated(position, vec2(-2.0, line_offset));
    vec3 sample_b = fetch_emulated(position, vec2(-1.0, line_offset));
    vec3 sample_c = fetch_emulated(position, vec2( 0.0, line_offset));
    vec3 sample_d = fetch_emulated(position, vec2( 1.0, line_offset));
    vec3 sample_e = fetch_emulated(position, vec2( 2.0, line_offset));

    float distance_x = distance_to_texel(position).x;
    float weight_a = gaussian_weight(distance_x - 2.0, bloom_pixel_softness);
    float weight_b = gaussian_weight(distance_x - 1.0, bloom_pixel_softness);
    float weight_c = gaussian_weight(distance_x,       bloom_pixel_softness);
    float weight_d = gaussian_weight(distance_x + 1.0, bloom_pixel_softness);
    float weight_e = gaussian_weight(distance_x + 2.0, bloom_pixel_softness);
    float weight_sum = max(weight_a + weight_b + weight_c + weight_d + weight_e, 1e-8);

    return (sample_a * weight_a + sample_b * weight_b + sample_c * weight_c +
            sample_d * weight_d + sample_e * weight_e) / weight_sum;
}

float scanline_weight(vec2 position, float line_offset)
{
    return gaussian_weight(distance_to_texel(position).y + line_offset, hard_scan);
}

float bloom_scanline_weight(vec2 position, float line_offset)
{
    return gaussian_weight(
        distance_to_texel(position).y + line_offset,
        bloom_scan_softness);
}

vec3 reconstruct_scanlines(vec2 position)
{
    vec3 upper  = horizontal_3(position, -1.0);
    vec3 center = horizontal_5(position,  0.0);
    vec3 lower  = horizontal_3(position,  1.0);

    float weight_upper  = scanline_weight(position, -1.0);
    float weight_center = scanline_weight(position,  0.0);
    float weight_lower  = scanline_weight(position,  1.0);

    return upper * weight_upper + center * weight_center + lower * weight_lower;
}

vec3 reconstruct_bloom(vec2 position)
{
    vec3 line_a = horizontal_5(position, -2.0);
    vec3 line_b = horizontal_7_bloom(position, -1.0);
    vec3 line_c = horizontal_7_bloom(position,  0.0);
    vec3 line_d = horizontal_7_bloom(position,  1.0);
    vec3 line_e = horizontal_5(position,  2.0);

    float weight_a = bloom_scanline_weight(position, -2.0);
    float weight_b = bloom_scanline_weight(position, -1.0);
    float weight_c = bloom_scanline_weight(position,  0.0);
    float weight_d = bloom_scanline_weight(position,  1.0);
    float weight_e = bloom_scanline_weight(position,  2.0);

    return line_a * weight_a + line_b * weight_b + line_c * weight_c +
           line_d * weight_d + line_e * weight_e;
}

vec2 warp_uv(vec2 position)
{
    position = position * 2.0 - 1.0;
    position *= vec2(
        1.0 + position.y * position.y * warp_x,
        1.0 + position.x * position.x * warp_y);
    return position * 0.5 + 0.5;
}

vec3 shadow_mask_value(vec2 pixel_position)
{
    vec3 mask = vec3(mask_dark);
    vec2 position = pixel_position * max(mask_scale, 0.0001);

    /* Very compressed TV style shadow mask. */
    if (shadow_mask == 1) {
        float line = mask_light;
        float odd = fract(position.x * (1.0 / 6.0)) < 0.5 ? 1.0 : 0.0;
        if (fract((position.y + odd) * 0.5) < 0.5) {
            line = mask_dark;
        }

        float column = fract(position.x * (1.0 / 3.0));
        if (column < 1.0 / 3.0) {
            mask.r = mask_light;
        }
        else if (column < 2.0 / 3.0) {
            mask.g = mask_light;
        }
        else {
            mask.b = mask_light;
        }
        mask *= line;
    }
    /* Aperture grille. */
    else if (shadow_mask == 2) {
        float column = fract(position.x * (1.0 / 3.0));
        if (column < 1.0 / 3.0) {
            mask.r = mask_light;
        }
        else if (column < 2.0 / 3.0) {
            mask.g = mask_light;
        }
        else {
            mask.b = mask_light;
        }
    }
    /* Stretched VGA style shadow mask. */
    else if (shadow_mask == 3) {
        position.x += position.y * 3.0;
        float column = fract(position.x * (1.0 / 6.0));
        if (column < 1.0 / 3.0) {
            mask.r = mask_light;
        }
        else if (column < 2.0 / 3.0) {
            mask.g = mask_light;
        }
        else {
            mask.b = mask_light;
        }
    }
    /* VGA style shadow mask. */
    else if (shadow_mask == 4) {
        position = floor(position * vec2(1.0, 0.5));
        position.x += position.y * 3.0;
        float column = fract(position.x * (1.0 / 6.0));
        if (column < 1.0 / 3.0) {
            mask.r = mask_light;
        }
        else if (column < 2.0 / 3.0) {
            mask.g = mask_light;
        }
        else {
            mask.b = mask_light;
        }
    }

    return mask;
}

void main()
{
    vec2 uv = gl_FragCoord.xy / output_size();
    vec2 warped_uv = warp_uv(uv);

    if (uv_outside(warped_uv)) {
        out_color = vec4(0.0);
        return;
    }

    vec3 color = reconstruct_scanlines(warped_uv);

    if (enable_bloom != 0 && bloom_amount > 0.0) {
        color += reconstruct_bloom(warped_uv) * bloom_amount;
    }

    if (shadow_mask > 0) {
        color *= shadow_mask_value(gl_FragCoord.xy * 1.000001);
    }

    float alpha = 1.0;
    if (preserve_alpha != 0) {
        alpha = sample_raw(warped_uv).a;
    }

    out_color = vec4(to_srgb(color), alpha);
}
