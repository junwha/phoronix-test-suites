/*
 *  pngfile.h - This file deals with PNG format image files (reading/writing)
 *
 *  $Id: pngfile.h,v 1.4 2017/04/22 22:01:11 johns Exp $
 */ 

/* read 24-bit RGB PNG */
int readpng(const char *name, int *xres, int *yres, unsigned char **imgdata);

/* write 24-bit RGB compressed PNG file */
int writepng(const char *name, int xres, int yres, unsigned char *imgdata);

/* write 32-bit RGBA compressed PNG file */
int writepng_alpha(const char *name, int xres, int yres, unsigned char *imgdata);



