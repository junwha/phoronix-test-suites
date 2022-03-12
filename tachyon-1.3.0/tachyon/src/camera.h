/*
 * camera.h - This file contains the defines for camera routines etc.
 *
 *  $Id: camera.h,v 1.27 2018/10/13 04:56:54 johns Exp $
 */

void camera_init(scenedef *);
void camray_init(scenedef *, ray *, unsigned long, unsigned long *, 
                 unsigned int, unsigned int);

void cameradefault(camdef *);
void cameraprojection(camdef *, int);
void cameradof(camdef *, flt focaldist, flt aperture);
void camerafrustum(camdef *, flt l, flt r, flt b, flt t);
void camerazoom(camdef *, flt zoom);
void cameraposition(camdef * camera, vector center, vector viewvec, 
                    vector upvec);
void getcameraposition(camdef * camera, vector * center, vector * viewvec, 
                       vector * upvec, vector *rightvec);

color cam_aa_perspective_ray(ray *, flt, flt);
void cam_prep_perspective_ray(ray *, flt, flt);
color cam_perspective_ray(ray *, flt, flt);

color cam_aa_dof_ray(ray *, flt, flt);
color cam_dof_ray(ray *, flt, flt);

color cam_aa_orthographic_ray(ray *, flt, flt);
color cam_orthographic_ray(ray *, flt, flt);

color cam_equirectangular_ray(ray *, flt, flt);
color cam_aa_equirectangular_ray(ray *, flt, flt);

color cam_stereo_equirectangular_ray(ray *, flt, flt);
color cam_stereo_aa_equirectangular_ray(ray *, flt, flt);

color cam_fisheye_ray(ray *, flt, flt);
color cam_aa_fisheye_ray(ray *, flt, flt);


