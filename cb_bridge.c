/* Batch independent CB posterior draws, with eta output and approximation disabled. */
#define CB_VALIDATE_ETA 1
#include "vendor/cb_core.c"
typedef struct { int pools, arms; cb_stream_t **stream; } project_bank;
void *project_create(int pools,int arms,uint64_t seed) {
    project_bank *b=calloc(1,sizeof(*b)); if(!b)return NULL;
    b->pools=pools;b->arms=arms;b->stream=calloc((size_t)pools*arms,sizeof(cb_stream_t*));
    if(!b->stream){free(b);return NULL;}
    for(int i=0;i<pools*arms;i++) {
        b->stream[i]=cb_stream_create(0.,0.,seed,(uint64_t)i,32);
        if(!b->stream[i]) {for(int j=0;j<i;j++)cb_stream_destroy(b->stream[j]);free(b->stream);free(b);return NULL;}
    }
    return b;
}
int project_update(void *ptr,const int *arms,const double *scores,const int *counts) {
    project_bank *b=ptr;
    for(int p=0;p<b->pools;p++) {
        int i=p*b->arms+arms[p];
        if(cb_stream_set_stats(b->stream[i],counts[p]*scores[p],counts[p])!=1)return -1;
    }
    return 0;
}
int project_choose(void *ptr,int *choices) {
    project_bank *b=ptr;
    for(int p=0;p<b->pools;p++) {
        double best=INFINITY;int choice=0;
        for(int k=0;k<b->arms;k++) {
            double sample;
            cb_stream_t *s=b->stream[p*b->arms+k];
            /* The core's point-family convention returns theta=0/1 even in
               diagnostic eta mode. Use the corresponding eta limits here. */
            if(s->nu>0 && s->chi==0.)sample=-INFINITY;
            else if(s->nu>0 && s->chi==s->nu)sample=INFINITY;
            else if(cb_stream_draw(s,1,&sample)!=1 || isnan(sample))return -1;
            if(k==0 || sample<best){best=sample;choice=k;}
        }
        choices[p]=choice;
    }
    return 0;
}
void project_destroy(void *ptr) {
    project_bank *b=ptr;if(!b)return;
    for(int i=0;i<b->pools*b->arms;i++)cb_stream_destroy(b->stream[i]);
    free(b->stream);free(b);
}
