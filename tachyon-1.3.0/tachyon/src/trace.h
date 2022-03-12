/* 
 * trace.h - This file contains the declarations for the main tracing calls.
 *
 *   $Id: trace.h,v 1.36 2013/06/13 06:18:28 johns Exp $
 */

typedef struct {
  int tid;                        /**< worker thread index            */
  int nthr;                       /**< total number of worker threads */
  scenedef * scene;               /**< scene handle                   */
  unsigned long * local_mbox;     /**< grid acceleration mailbox structure */
  unsigned long serialno;         /**< ray mailbox test serial number */
  int startx;                     /**< starting X pixel index         */
  int stopx;                      /**< ending X pixel index           */
  int xinc;                       /**< X pixel stride                 */
  int starty;                     /**< starting Y pixel index         */
  int stopy;                      /**< ending Y pixel index           */
  int yinc;                       /**< Y pixel stride                 */
  rt_barrier_t * runbar;          /**< sleeping thread pool barrier   */
#if defined(THR)
  int sched_dynamic;              /**< pixel scheduler mode           */
  rt_atomic_int_t * pixelsched;   /**< atomic pixel scheduler counter */
#endif
#if defined(MPI) && defined(THR)
  int numrowbars;                 /**< number of row barriers         */
  rt_atomic_int_t * rowbars;      /**< per-row atomic int barriers    */
  rt_atomic_int_t * rowsdone;     /**< counter of rows completed      */
#endif
} thr_parms;

color trace(ray *);
void * thread_trace(thr_parms *); 

