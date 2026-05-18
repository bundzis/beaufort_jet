/*
**
** Options for 3D baroclinic Beaufort JET - Idealized ice w/ bulk fluxes + DVD
*/

#define ROMS_MODEL
#define BEAUFORT_JET_ICE_W_DVD

/* Basic dynamics */
#define UV_ADV
#define UV_COR
#define UV_LOGDRAG

#define SALINITY
#define SOLVE3D
#define SPLINES_VVISC
#define SPLINES_VDIFF
#define ASSUMED_SHAPE

#undef SPLINES
#undef MASKING

#define AVERAGES

/* Turbulence Closure/Vertical Mixing */
#define GLS_MIXING
#ifdef GLS_MIXING
# define CANUTO_A  /* OG */
# undef CANUTO_B   /* For Testing */
# define N2S2_HORAVG
#endif

/* Large-McWilliams-Doney K-Profile Parameterization */
#undef LMD_MIXING       
#ifdef LMD_MIXING 
# define LMD_SHAPIRO      /* Shapiro filter for vertical mixing */
# define LMD_RIMIX        /* Richardson number dependent mixing */
# define LMD_CONVEC       /* Convective adjustment */
# define LMD_SKPP         /* Surface KPP scheme */
# define LMD_BKPP         /* Bottom KPP scheme */
# define LMD_NONLOCAL     /* Nonlocal transport */ 
# define RI_HORAVG
# define RI_VERAVG
#endif

/* Horizontal mixing */
#define UV_VIS2
#define TS_DIF2
#define MIX_S_UV
#define MIX_GEO_TS

/* Analytic surface/bottom fluxes */
#define BULK_FLUXES
#define ANA_WINDS
#define ANA_BTFLUX
#define ANA_BSFLUX
#define ANA_FSOBC
#define ANA_M2OBC

/* ANA files do not work as well as forcing files */
#undef ANA_CLOUD
#undef ANA_TAIR
#undef ANA_HUMIDITY
#undef ANA_RAIN
#undef ANA_PAIR
#undef ANA_SRFLUX

#define EMINUSP
#define LONGWAVE_OUT
#undef SHORTWAVE
#undef ALBEDO

/* Analytic climatology and BC options */
#define ANA_NUDGCOEF
#define ANA_FSOBC
#undef ANA_TOBC
#undef ANA_M2CLIMA
#define ANA_M3CLIMA
#define ANA_M2OBC     
#define ANA_M3OBC     

/* Sea Ice */
#define ICE_MODEL
#ifdef ICE_MODEL
# define ICE_BULK_FLUXES
# define ICE_THERMO
# define ICE_MK
# define ICE_ALBEDO
# define ICE_ALB_EC92
# define ICE_MOMENTUM
# define ICE_EVP
# define ICE_ADVECT
# define ICE_SMOLAR
# define ICE_UPWIND
# define ICE_CONVSNOW
#endif

/* DVD Options */
#define TS_VAR
#define T_PASSIVE
#define ANA_PASSIVE
#define ANA_BPFLUX        
#define ANA_SPFLUX


