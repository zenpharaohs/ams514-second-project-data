#define CB_VALIDATE_ETA 1
#include "vendor/cb_core.c"
typedef struct {int pools,arms;cb_stream_t **s;} bank;
void *bank_create(int pools,int arms,uint64_t seed){
    bank *b=calloc(1,sizeof(*b));if(!b)return NULL;b->pools=pools;b->arms=arms;
    b->s=calloc(pools*arms,sizeof(cb_stream_t*));if(!b->s){free(b);return NULL;}
    for(int i=0;i<pools*arms;i++){b->s[i]=cb_stream_create(0,0,seed,i,32);if(!b->s[i])return NULL;}
    return b;
}
int bank_update(void *ptr,const int *arms,const double *z,const double *nu){
    bank *b=ptr;for(int p=0;p<b->pools;p++)if(cb_stream_set_stats(b->s[p*b->arms+arms[p]],z[p]*nu[p],nu[p])!=1)return -1;return 0;
}
int bank_reset(void *ptr,int pool,const double *z,const double *nu){
    bank *b=ptr;for(int a=0;a<b->arms;a++)if(cb_stream_set_stats(b->s[pool*b->arms+a],z[a]*nu[a],nu[a])!=1)return -1;return 0;
}
int bank_choose(void *ptr,int *out){
    bank *b=ptr;for(int p=0;p<b->pools;p++){
        double best=INFINITY;int winner=0;
        for(int a=0;a<b->arms;a++){
            cb_stream_t *s=b->s[p*b->arms+a];double x;
            if(s->nu>0&&s->chi==0)x=-INFINITY;
            else if(s->nu>0&&s->chi==s->nu)x=INFINITY;
            else if(cb_stream_draw(s,1,&x)!=1||isnan(x))return -1;
            if(a==0||x<best){best=x;winner=a;}
        }out[p]=winner;
    }return 0;
}
void bank_destroy(void *ptr){bank*b=ptr;if(!b)return;for(int i=0;i<b->pools*b->arms;i++)cb_stream_destroy(b->s[i]);free(b->s);free(b);}
int diagnostic_draw(const double *z,const double *nu,int k,int n,uint64_t seed,double *out){
    double *buffer=malloc(n*sizeof(double));if(!buffer)return -1;
    for(int a=0;a<k;a++){
        if(z[a]==0||z[a]==1){for(int j=0;j<n;j++)out[j*k+a]=z[a]==0?-INFINITY:INFINITY;continue;}
        cb_stream_t*s=cb_stream_create(z[a]*nu[a],nu[a],seed,a,32);if(!s){free(buffer);return -2;}
        if(cb_stream_draw(s,n,buffer)!=n){cb_stream_destroy(s);free(buffer);return -3;}
        for(int j=0;j<n;j++)out[j*k+a]=buffer[j];cb_stream_destroy(s);
    }free(buffer);return 0;
}
