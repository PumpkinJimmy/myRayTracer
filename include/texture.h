#pragma once

#ifndef TEXTURE_H
#define TEXTURE_H
#define STB_IMAGE_IMPLEMENTATION
#include "ray.h"
#include "stb_image.h"
#include <filesystem>

namespace fs = std::filesystem;

class texture {
public:
    virtual color value(double u, double v, const point3& p) const = 0;
};

class solid_color : public texture {
public:
    solid_color() {}
    solid_color(color c) : color_value(c) {}

    solid_color(double red, double green, double blue)
        : solid_color(color(red, green, blue)) {}

    virtual color value(double u, double v, const vec3& p) const override {
        return color_value;
    }

private:
    color color_value;
};


class checker_texture : public texture {
public:
    checker_texture() {}
    checker_texture(shared_ptr<texture> _even, shared_ptr<texture> _odd)
        : even(_even), odd(_odd) {}
    checker_texture(color c1, color c2)
        : even(make_shared<solid_color>(c1)), odd(make_shared<solid_color>
            (c2)) {}
    virtual color value(double u, double v, const point3& p) const override
    {
        auto sines = sin(10 * p.x()) * sin(10 * p.y()) * sin(10 * p.z());
        if (sines < 0)
            return odd->value(u, v, p);
        else
            return even->value(u, v, p);
    }
public:
    shared_ptr<texture> odd;
    shared_ptr<texture> even;
};

class ImageTexture : public texture {
public:
    const static int bytes_per_pixel = 3;
    typedef shared_ptr<ImageTexture> Ptr;
    typedef shared_ptr<const ImageTexture> ConstPtr;

    ImageTexture()
        : data(nullptr), width(0), height(0), bytes_per_scanline(0) {}
    ImageTexture(const char* filename) {
        if (!fs::is_regular_file(filename)) {
            std::cerr << "ERROR: No such image file: " << filename << std::endl;
        }
        auto components_per_pixel = bytes_per_pixel;

        data = stbi_load(
            filename, &width, &height, &components_per_pixel,
            components_per_pixel);
        if (!data) {
            std::cerr << "ERROR: Could not load texture image file " << filename << ".\n";
        }
        bytes_per_scanline = bytes_per_pixel * width;
    }
    


    template <typename... Args>
    static ImageTexture::Ptr create(Args... args) {
        return make_shared<ImageTexture>(args...);
    }

    ~ImageTexture() {
        delete data;
    }
    static double adjustTexCoord(double x) {
        double f = x - int(x);
        return f < 0 ? f + 1 : f;
    }

    virtual color value(double u, double v, const vec3& p) const override {
        if (data == nullptr) {
            return color(0, 1, 1);
        }

        u = adjustTexCoord(u);
        v = 1.0 - adjustTexCoord(v);

        auto i = static_cast<int>(u * width);
        auto j = static_cast<int>(v * height);

        if (i >= width) i = width - 1;
        if (j >= height) j = height - 1;

        const auto color_scale = 1.0 / 255.0;
        auto pixel = data + j * bytes_per_scanline + i * bytes_per_pixel;

        return color(
            color_scale * pixel[0],
            color_scale * pixel[1],
            color_scale * pixel[2]);
        
    }
private:
    unsigned char* data;
    int width, height;
    int bytes_per_scanline;
};

#endif