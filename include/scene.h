#pragma once
#ifndef _SCENE_H
#define _SCENE_H
#include "common.h"
#include "hittable_list.h"
#include "hittable.h"
#include "sphere.h"
#include "material.h"
#include "texture.h"
#include "aarect.h"

HittableList random_scene();
HittableList simple_light();
HittableList cornell_box();

#endif