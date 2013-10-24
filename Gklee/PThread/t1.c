/******************************************************************
 * simple.c -- multithreaded "hello world"
 */
/* Linux with glibc:
 *   _REENTRANT to grab thread-safe libraries
 *   _POSIX_SOURCE to get POSIX semantics
 */

#ifdef __linux__
#define _REENTRANT
#define _POSIX_SOURCE
#endif

/* Hack for LinuxThreads */
#ifdef __linux__
#define _P __P
#endif

#include <pthread.h>
#include <string.h>   /* for strerror() */
#include <stdio.h>
#include <stdlib.h>

#define NTHREADS 4
 
#define errexit(code,str)			    \
  fprintf(stderr,"%s: %s\n",(str),strerror(code));  \
  exit(1);


/******** this is the thread code */
void *hola(void * arg) {
  int myid=*(int *) arg;
  
  printf("Hello, world, I'm %d\n",myid);
  return arg;
}

/******** this is the main thread's code */
int main(int argc,char *argv[]) {
  int worker;
  pthread_t threads[NTHREADS];                /* holds thread info */
  int ids[NTHREADS];                          /* holds thread args */
  int errcode;                                /* holds pthread error code */
  int *status;                                /* holds return code */
  
  /* create the threads */
  for (worker=0; worker<NTHREADS; worker++) {
    ids[worker]=worker;
    if (errcode=pthread_create(&threads[worker],/* thread struct             */
			       NULL,                    /* default thread attributes */
			       hola,                    /* start routine             */
			       &ids[worker])) {         /* arg to routine            */
      errexit(errcode,"pthread_create");
    }
  }
  /* reap the threads as they exit */
  for (worker=0; worker<NTHREADS; worker++) {
    /* wait for thread to terminate */
    if (errcode=pthread_join(threads[worker],(void *) &status)) { 
      errexit(errcode,"pthread_join");
    }
    /* check thread's exit status and release its resources */
    if (*status != worker) {
      fprintf(stderr,"thread %d terminated abnormally\n",worker);
      exit(1);
    }
  }
  return(0);
}

/* EOF simple.c */
