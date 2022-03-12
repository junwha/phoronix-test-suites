/* 
 * sphere.h - This file contains the defines for spheres etc.
 *
 *  $Id: sphere.h,v 1.14 2018/08/28 15:17:20 johns Exp $
 */

object * newsphere(void *, vector, flt);

#ifdef SPHERE_PRIVATE

typedef struct {
  RT_OBJECT_HEAD
  vector ctr;    /**< sphere center */
  flt rad;       /**< sphere radius */
} sphere;

static int sphere_bbox(void * obj, vector * min, vector * max);
static void sphere_intersect(const sphere *, ray *);
static void sphere_normal(const sphere *, const vector *, const ray *, vector *);

#endif /* SPHERE_PRIVATE */

