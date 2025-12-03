/*
**
** Options for 3D baroclinic Beaufort JET - Idealized ice w/ bulk fluxes + DVD
*/

#define ROMS_MODEL
#define BEAUFORT_JET_ICE_BULK_FLUXES_W_DVD

/* Basic dynamics */
#define UV_ADV
#define UV_COR
#define UV_LOGDRAG

#define SALINITY
#define SOLVE3D
#define SPLINES_VVISC
#define SPLINES_VDIFF
#undef SPLINES
#undef MASKING

#define AVERAGES

/* Turbulence */
#define GLS_MIXING
#define CANUTO_A
#define N2S2_HORAVG

/* Analytic surface/bottom fluxes */
#define ANA_BTFLUX
#define ANA_BSFLUX
#define ANA_CLOUD
#define ANA_TAIR
#define ANA_HUMIDITY
#define ANA_WINDS
#define ANA_RAIN
#define ANA_PAIR
#define ANA_SRFLUX

#define EMINUSP
#define ALBEDO
#define LONGWAVE
#define SHORTWAVE

/* Analytic climatology and BC options */
#define ANA_M3CLIMA
#define ANA_NUDGCOEF
#define ANA_FSOBC
#define ANA_TOBC
#define ANA_M2CLIMA
#define ANA_M2OBC     
#define ANA_M3OBC     

#undef UV_VIS2
#undef VISC_GRID
#undef ASSUMED_SHAPE

/* Sea Ice */
#define ICE_MODEL
#ifdef ICE_MODEL
# define BULK_FLUXES
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


