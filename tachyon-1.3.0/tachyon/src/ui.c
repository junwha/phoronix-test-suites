/*
 * ui.c - Contains functions for dealing with user interfaces
 *
 *  $Id: ui.c,v 1.10 2013/06/14 16:49:10 johns Exp $
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define TACHYON_INTERNAL 1
#include "tachyon.h"
#include "macros.h"
#include "util.h"
#include "ui.h"
#include "global.h"

void rt_set_ui_message(void (* func) (int, char *)) {
  global_rt_ui_message = func;
}

void rt_set_ui_progress(void (* func) (int)) {
  global_rt_ui_progress = func;
}

void rt_ui_message(int level, char * msg) {
  if (global_rt_ui_message != NULL) 
    global_rt_ui_message(level, msg);
}

void rt_ui_progress(int percent) {
  if (global_rt_ui_progress != NULL)
    global_rt_ui_progress(percent);
}

int rt_ui_checkaction(void) {
  if (global_rt_ui_checkaction != NULL) 
    return global_rt_ui_checkaction();
  else
    return 0;
}














