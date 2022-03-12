/*
 * global.c - any/all global data items etc should be in this file 
 *
 *  $Id: global.c,v 1.20 2013/06/14 16:49:10 johns Exp $
 *
 */

#define TACHYON_INTERNAL 1
#include <stdlib.h>
#include "tachyon.h"
#include "parallel.h"

rt_parhandle global_parhnd = NULL;  /**< parallel message passing data structures */

rawimage * global_imagelist[MAXIMGS]; /**< texture map cache */
int global_numimages;

void (* global_rt_ui_message) (int, char *) = NULL;
void (* global_rt_ui_progress) (int) = NULL;
int (* global_rt_ui_checkaction) (void) = NULL;

