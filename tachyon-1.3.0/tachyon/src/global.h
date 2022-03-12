/*
 * global.h - any/all global data items etc should be in this file
 *
 *  $Id: global.h,v 1.20 2013/06/14 16:49:10 johns Exp $
 *
 */

extern rt_parhandle global_parhnd;  /**< parallel message passing data structures */
extern rawimage * global_imagelist[MAXIMGS]; /**< texture map cache */
extern int global_numimages;

extern void (* global_rt_ui_message) (int, char *);
extern void (* global_rt_ui_progress) (int);
extern int (* global_rt_ui_checkaction) (void);


