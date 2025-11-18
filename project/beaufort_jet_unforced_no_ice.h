/*
**
** Options for 3D baroclinic beaufort JET
**
** Application flag:   BEAUFORT_JET
** Input scripts:      ocean_beaufort_2020_dvd_myroms_ice.in
**                     ice.in
*/

#define ROMS_MODEL
#define BEAUFORT_JET_UNFORCED_NO_ICE

#define UV_ADV
#define UV_COR
#define UV_LOGDRAG

#define SALINITY
#define SOLVE3D
#define SPLINES_VVISC
#define SPLINES_VDIFF
#undef SPLINES
#undef MASKING

#undef AVERAGES

#define ANA_M2CLIMA
#define ANA_M2OBC
#define M3CLIMATOLOGY
#define M3CLM_NUDGING
#define ANA_M3CLIMA
#define ANA_NUDGCOEF

#define GLS_MIXING
#define CANUTO_A
#define N2S2_HORAVG

#define ANA_BSFLUX
#define ANA_BTFLUX
#define ANA_FSOBC
#define ANA_SMFLUX
#define ANA_SSFLUX
#define ANA_STFLUX

#undef UV_VIS2
#undef VISC_GRID
#undef ASSUMED_SHAPE