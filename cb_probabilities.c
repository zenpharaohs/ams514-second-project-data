/* Deterministic winner probabilities and d/d log(nu), CB conjugate family.
   Integrate each sign of eta in v=log|eta|. No random draws in fitting. */
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct { double lo,h,mass,dmass; double *f,*dv,*cdf,*dcdf,*tail,*dtail; } branch;
static int compare_double(const void *a,const void *b){double x=*(const double*)a,y=*(const double*)b;return (x>y)-(x<y);}

static double H(double a) {
    if(a<.1) {double a2=a*a;return a*.5+a2*(-1./24.+a2*(1./2880.+a2*(-1./181440.+a2/9676800.)));}
    return log(a)-log(-expm1(-a));
}
static double D(double a) {
    if(a<.1) {double a2=a*a;return a*.5+a2*(-1./12.+a2*(1./720.+a2*(-1./30240.+a2/1209600.)));}
    return a>700 ? 1 : 1-a/expm1(a);
}
static double g(double v,double w,double nu) {double a=exp(v);return nu*(H(a)-w*a);}
static double log_density(double v,double w,double nu) {return v+g(v,w,nu);}
static double slope(double v,double w,double nu) {double a=exp(v);return 1+nu*(D(a)-w*a);}

static void bounds(double w,double nu,double *left,double *right,double *mode) {
    double hi=log1p(1/nu)-log(w)+1,lo=fmin(-35-log1p(nu),hi-40);
    for(int i=0;i<60;i++){double mid=(lo+hi)*.5;if(slope(mid,w,nu)>0)lo=mid;else hi=mid;}
    *mode=(lo+hi)*.5;double peak=log_density(*mode,w,nu);
    lo=*mode-60;hi=*mode;
    for(int i=0;i<45;i++){double mid=(lo+hi)*.5;if(log_density(mid,w,nu)<peak-42)lo=mid;else hi=mid;}
    *left=(lo+hi)*.5;lo=*mode;hi=*mode+10;
    for(int i=0;i<45;i++){double mid=(lo+hi)*.5;if(log_density(mid,w,nu)>peak-42)lo=mid;else hi=mid;}
    *right=(lo+hi)*.5;
}
static double hermite(double y0,double y1,double d0,double d1,double u,double h) {
    double u2=u*u,u3=u2*u;
    return (2*u3-3*u2+1)*y0+(u3-2*u2+u)*h*d0+(-2*u3+3*u2)*y1+(u3-u2)*h*d1;
}
static void survival(branch *b,int n,double v,int negative,double *s,double *ds) {
    double position=(v-b->lo)/b->h;
    if(position<=0){*s=negative?0:b->mass;*ds=negative?0:b->dmass;return;}
    if(position>=n-1){*s=negative?b->mass:0;*ds=negative?b->dmass:0;return;}
    int i=(int)position;double u=position-i,sign=negative?1:-1;
    double *c=negative?b->cdf:b->tail,*dc=negative?b->dcdf:b->dtail;
    *s=hermite(c[i],c[i+1],sign*b->f[i],sign*b->f[i+1],u,b->h);
    *ds=hermite(dc[i],dc[i+1],sign*b->dv[i],sign*b->dv[i+1],u,b->h);
    if(*s<0)*s=0;if(*s>b->mass)*s=b->mass;
}

int cb_winner(const double *z,const double *nu,int k,int n,double *p,double *jac) {
    if(k<1||k>64||n<33||n%2!=1)return -1;
    memset(p,0,k*sizeof(double));memset(jac,0,k*k*sizeof(double));
    for(int i=0;i<k;i++)if(z[i]<=0){p[i]=1;return 0;}
    branch *b=calloc(2*k,sizeof(branch));double *mem=calloc((size_t)2*k*n*6,sizeof(double));
    double *tmp=malloc((size_t)4*n*sizeof(double));
    double lognorm[64],gmean[64];
    if(!b||!mem||!tmp){free(b);free(mem);free(tmp);return -2;}
    for(int i=0;i<2*k;i++){
        double *start=mem+(size_t)i*n*6;
        b[i].f=start;b[i].dv=start+n;b[i].cdf=start+2*n;b[i].dcdf=start+3*n;b[i].tail=start+4*n;b[i].dtail=start+5*n;
    }
    for(int arm=0;arm<k;arm++){
        if(!(z[arm]>0&&z[arm]<1&&nu[arm]>0)){free(b);free(mem);free(tmp);return -3;}
        double peak[2],mode[2],right[2],gref=0;
        for(int sign=0;sign<2;sign++){
            double w=sign?1-z[arm]:z[arm];branch *q=&b[2*arm+sign];
            bounds(w,nu[arm],&q->lo,&right[sign],&mode[sign]);q->h=(right[sign]-q->lo)/(n-1);
            peak[sign]=log_density(mode[sign],w,nu[arm]);
        }
        int major=peak[1]>peak[0];double maximum=fmax(peak[0],peak[1]);
        gref=g(mode[major],major?1-z[arm]:z[arm],nu[arm]);
        double total=0,first=0;
        for(int sign=0;sign<2;sign++){
            branch *q=&b[2*arm+sign];double w=sign?1-z[arm]:z[arm];
            double *deriv=tmp,*dderiv=tmp+n;
            for(int j=0;j<n;j++){
                double v=q->lo+j*q->h,gg=g(v,w,nu[arm])-gref,d=slope(v,w,nu[arm]);
                q->f[j]=exp(v+gg+gref-maximum);q->dv[j]=q->f[j]*gg;
                deriv[j]=q->f[j]*d;dderiv[j]=q->f[j]*(gg*d+d-1);
            }
            for(int j=1;j<n;j++){
                double mass=q->h*.5*(q->f[j-1]+q->f[j])+q->h*q->h/12*(deriv[j-1]-deriv[j]);
                double dm=q->h*.5*(q->dv[j-1]+q->dv[j])+q->h*q->h/12*(dderiv[j-1]-dderiv[j]);
                if(mass<0)mass=0;
                q->cdf[j]=q->cdf[j-1]+mass;q->dcdf[j]=q->dcdf[j-1]+dm;
            }
            q->mass=q->cdf[n-1];q->dmass=q->dcdf[n-1];total+=q->mass;first+=q->dmass;
        }
        if(!(total>0&&isfinite(total))){free(b);free(mem);free(tmp);return -4;}
        double eg=first/total;
        lognorm[arm]=maximum+log(total);gmean[arm]=gref+eg;
        for(int sign=0;sign<2;sign++){
            branch *q=&b[2*arm+sign];
            for(int j=0;j<n;j++){
                q->dv[j]=(q->dv[j]-eg*q->f[j])/total;q->f[j]/=total;
                q->dcdf[j]=(q->dcdf[j]-eg*q->cdf[j])/total;q->cdf[j]/=total;
            }
            q->dmass=q->dcdf[n-1];q->mass=q->cdf[n-1];
            // Accumulate right tails directly, retaining small survival masses.
            for(int j=n-2;j>=0;j--){
                double v0=q->lo+j*q->h,v1=v0+q->h;
                double w=sign?1-z[arm]:z[arm],d0=slope(v0,w,nu[arm]),d1=slope(v1,w,nu[arm]);
                double mass=q->h*.5*(q->f[j]+q->f[j+1])+q->h*q->h/12*(q->f[j]*d0-q->f[j+1]*d1);
                double dm=q->h*.5*(q->dv[j]+q->dv[j+1])+q->h*q->h/12*(q->dv[j]*d0+q->f[j]*(d0-1)-q->dv[j+1]*d1-q->f[j+1]*(d1-1));
                q->tail[j]=q->tail[j+1]+fmax(0,mass);q->dtail[j]=q->dtail[j+1]+dm;
            }
        }
    }
    double *knots=malloc((size_t)k*n*sizeof(double));
    if(!knots){free(b);free(mem);free(tmp);return -6;}
    const double gauss[3]={-.7745966692414834,0,.7745966692414834};
    const double weights[3]={5./9.,8./9.,5./9.};
    // Every arm's mesh enters the common mesh: a broad arm's integral must
    // resolve a narrow competitor's CDF transition as well as its own density.
    for(int sign=0;sign<2;sign++){
        for(int i=0;i<k;i++)for(int u=0;u<n;u++)knots[i*n+u]=b[2*i+sign].lo+u*b[2*i+sign].h;
        qsort(knots,k*n,sizeof(double),compare_double);
        for(int u=1;u<k*n;u++){
            double half=(knots[u]-knots[u-1])*.5;if(half<=0)continue;
            double mid=(knots[u]+knots[u-1])*.5;
            for(int node=0;node<3;node++){
                double v=mid+half*gauss[node],weight=half*weights[node];
                double s[64],ds[64],f[64],df[64];
                for(int j=0;j<k;j++){
                    survival(&b[2*j+sign],n,v,!sign,&s[j],&ds[j]);
                    if(!sign){s[j]+=b[2*j+1].mass;ds[j]+=b[2*j+1].dmass;}
                    double gg=g(v,sign?1-z[j]:z[j],nu[j]);
                    f[j]=exp(v+gg-lognorm[j]);df[j]=f[j]*(gg-gmean[j]);
                }
                for(int i=0;i<k;i++){
                    double product=1;for(int j=0;j<k;j++)if(j!=i)product*=s[j];
                    p[i]+=weight*f[i]*product;jac[i*k+i]+=weight*df[i]*product;
                    for(int j=0;j<k;j++)if(j!=i){
                        double other;
                        if(s[j]>1e-100)other=product/s[j];
                        else {other=1;for(int a=0;a<k;a++)if(a!=i&&a!=j)other*=s[a];}
                        jac[i*k+j]+=weight*f[i]*other*ds[j];
                    }
                }
            }
        }
    }
    free(knots);
    double sum=0;double jsum[64]={0};
    for(int i=0;i<k;i++){sum+=p[i];for(int j=0;j<k;j++)jsum[j]+=jac[i*k+j];}
    if(!(sum>0&&isfinite(sum))){free(b);free(mem);free(tmp);return -5;}
    for(int i=0;i<k;i++){p[i]/=sum;for(int j=0;j<k;j++)jac[i*k+j]=(jac[i*k+j]-p[i]*jsum[j])/sum;}
    free(b);free(mem);free(tmp);return 0;
}
