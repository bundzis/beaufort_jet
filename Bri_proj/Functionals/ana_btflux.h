      SUBROUTINE ana_btflux (ng, tile, model, itrc)
!
!! svn $Id$
!!======================================================================
!! Copyright (c) 2002-2014 The ROMS/TOMS Group                         !
!!   Licensed under a MIT/X style license                              !
!!   See License_ROMS.txt                                              !
!=======================================================================
!                                                                      !
!  This routine sets kinematic bottom flux of tracer type variables    !
!  (tracer units m/s).                                                 !
!                                                                      !
!=======================================================================
!
      USE mod_param
      USE mod_forces
      USE mod_ncparam
#ifdef BH_BOTTOM_RESP
!      USE mod_param
      USE mod_ocean
      USE mod_stepping
#endif
!
! Imported variable declarations.
!
      integer, intent(in) :: ng, tile, model, itrc

#include "tile.h"
!
      CALL ana_btflux_tile (ng, tile, model, itrc,                      &
     &                      LBi, UBi, LBj, UBj,                         &
#ifdef BH_BOTTOM_RESP
     &                      nnew(ng),nstp(ng),                          &
     &                      OCEAN(ng)%t,                                &
#endif
     &                      IminS, ImaxS, JminS, JmaxS,                 &
#ifdef TL_IOMS
     &                      FORCES(ng) % tl_btflx,                      &
#endif
     &                      FORCES(ng) % btflx)
!
! Set analytical header file name used.
!
#ifdef DISTRIBUTE
      IF (Lanafile) THEN
#else
      IF (Lanafile.and.(tile.eq.0)) THEN
#endif
        ANANAME( 3)=__FILE__
      END IF

      RETURN
      END SUBROUTINE ana_btflux
!
!***********************************************************************
      SUBROUTINE ana_btflux_tile (ng, tile, model, itrc,                &
     &                            LBi, UBi, LBj, UBj,                   &
#ifdef BH_BOTTOM_RESP
     &                            nnew,nstp,                            &
     &                            t,                                    &
#endif
     &                            IminS, ImaxS, JminS, JmaxS,           &
#ifdef TL_IOMS
     &                            tl_btflx,                             &
#endif
     &                            btflx)
!***********************************************************************
!
      USE mod_param
      USE mod_scalars
!
!  Imported variable declarations.
!
      integer, intent(in) :: ng, tile, model, itrc
      integer, intent(in) :: LBi, UBi, LBj, UBj
      integer, intent(in) :: IminS, ImaxS, JminS, JmaxS
!
#ifdef BH_BOTTOM_RESP
      integer, intent(in) :: nnew, nstp
      real(r8), intent(inout) :: t(LBi:,LBj:,:,:,:)
!      real(r8), external :: O2_sat, O2_sat_BK
#endif
#ifdef ASSUMED_SHAPE
      real(r8), intent(inout) :: btflx(LBi:,LBj:,:)
# ifdef TL_IOMS
      real(r8), intent(inout) :: tl_btflx(LBi:,LBj:,:)
# endif
#else
      real(r8), intent(inout) :: btflx(LBi:UBi,LBj:UBj,NT(ng))
# ifdef TL_IOMS
      real(r8), intent(inout) :: tl_btflx(LBi:UBi,LBj:UBj,NT(ng))
# endif
#endif
!
!  Local variable declarations.
!
      integer :: i, j, k

#include "set_bounds.h"
!
!-----------------------------------------------------------------------
!  Set kinematic bottom heat flux (degC m/s) at horizontal RHO-points.
!-----------------------------------------------------------------------
!
      IF (itrc.eq.itemp) THEN
        DO j=JstrT,JendT
          DO i=IstrT,IendT
            btflx(i,j,itrc)=0.0_r8
#ifdef TL_IOMS
            tl_btflx(i,j,itrc)=0.0_r8
#endif
          END DO
        END DO
!
!-----------------------------------------------------------------------
!  Set kinematic bottom salt flux (m/s) at horizontal RHO-points,
!  scaling by bottom salinity is done elsewhere.
!-----------------------------------------------------------------------
!
      ELSE IF (itrc.eq.isalt) THEN
        DO j=JstrT,JendT
          DO i=IstrT,IendT
            btflx(i,j,itrc)=0.0_r8
#ifdef TL_IOMS
            tl_btflx(i,j,itrc)=0.0_r8
#endif
          END DO
        END DO

#ifdef T_PASSIVE  !Added by DJ
!
!-----------------------------------------------------------------------
!  Set kinematic bottom flux (T m/s) of passive tracers, if any.
!-----------------------------------------------------------------------
!
      ELSE IF (itrc.eq.inert(1)) THEN

!     Bottom flux of O2 is actually done as a tracer boundary condition in
!     analytical.F.  This code only sets the surface to saturation and clips
!     negative oxigen values
!      print*,'zzzzzzzzz',Istr,Iend,LBi, UBi, LBj, UBj
!      print*, 'itemp',itemp,'inert1', inert(1),'inert2',inert(2)

# ifdef BH_BOTTOM_RESP
       DO j=JstrT,JendT
          DO i=IstrT,IendT
!             t(i,j,N(ng),nstp,inert(1)) = 1000000
             t(i,j,N(ng),nstp,inert(1))=O2_sat( t(i,j,N(ng),nstp,itemp), &
    &                                           t(i,j,N(ng),nstp,isalt) )     
!             t(i,j,N(ng),nstp,inert(1))=O2_sat_BK( t(i,j,N(ng),nstp,itemp), &
!     &                                             t(i,j,N(ng),nstp,isalt) )     
          END DO
       END DO
       DO k=1,N(ng)
          DO j=JstrT,JendT
             DO i=IstrT,IendT
                t(i,j,k,nstp,inert(1))=MAX(t(i,j,k,nstp,inert(1)),0.0_r8)
             END DO
          END DO
       END DO
# endif
       DO j=JstrR,JendR
          DO i=IstrR,IendR
# if defined BH_BOTTOM_RESP
            if (itrc.eq.inert(1)) then
               if(t(i,j,1,nstp,itrc).gt.0.0) then
                   btflx(i,j,itrc)=6.0*2.0**(0.1*t(i,j,1,nstp,itemp))    &
     &                            *(1.0-exp(-0.033*t(i,j,1,nstp,inert(1))))/86400.0
!                  write(*,*) 'itrc=', itrc
!                  write(*,*) 'btflx=', btflx(i,j,itrc)
               else 
                  btflx(i,j,itrc)=0.0_r8
	       endif
            else
                btflx(i,j,itrc)=0.0_r8
            endif
# else
            btflx(i,j,itrc)=0.0_r8
# endif
# ifdef TL_IOMS
            tl_btflx(i,j,itrc)=0.0_r8
# endif
          END DO
       END DO
#endif
      END IF

      RETURN
      END SUBROUTINE ana_btflux_tile


      FUNCTION O2_sat (T, S)
!
!=======================================================================
!                                                                      !
!  This routine computes the saturation value of oxygen                !
!  in the surface seawater.                                            !
!                                                                      !
!  On Input:                                                           !
!                                                                      !
!     T          Temperature (Celsius).                                !
!     S          Salinity (PSS).                                       !
!                                                                      !
!  On Output:                                                          !
!                                                                      !
!     O2_sat     Saturation value of O2 (mmol / m3).                   !
!======================================================================!
!  Algorithm based on Weiss (1970)                                     !
!=======================================================================
!
      USE mod_kinds
!
!  Function result
!
      real(r8) :: O2_sat
!
!  Imported variable declarations.
!
      real(r8), intent(in) :: T
      real(r8), intent(in) :: S
!
!  Local variable declarations.
!
      real(r8), parameter :: A1 = -173.4292
      real(r8), parameter :: A2 =  249.6339
      real(r8), parameter :: A3 =  143.3483
      real(r8), parameter :: A4 =  -21.8492
      real(r8), parameter :: B1 =   -0.033096
      real(r8), parameter :: B2 =    0.014259
      real(r8), parameter :: B3 =   -0.0017000
      real(r8) :: Tk
!
!     Convert temperature to Deg. Kelvin
!
      Tk = T + 273.15
!
!     Calculate saturation O2.
!
!      O2_sat = EXP(A1+A2*(100.0/Tk)+A3*LOG(Tk/100.0)+A4*(Tk/100.0) +    &
!     &     S*(B1+B2*(Tk/100.0)+B3*((Tk/100.0)**2)) )*(44.661*1.42903)

      O2_sat = EXP(A1+A2*(100.0/Tk)+A3*LOG(Tk/100.0)+A4*(Tk/100.0) +    &
     &     S*(B1+B2*(Tk/100.0)+B3*((Tk/100.0)**2)) )*(44.661)
      RETURN
      END FUNCTION O2_sat


      FUNCTION O2_sat_BK (T, S)
!
!=======================================================================
!                                                                      !
!  This routine computes the saturation value of oxygen                !
!  in the surface seawater.                                            !
!                                                                      !
!  On Input:                                                           !
!                                                                      !
!     T          Temperature (Celsius).                                !
!     S          Salinity (PSS).                                       !
!                                                                      !
!  On Output:                                                          !
!                                                                      !
!     O2_sat     Saturation value of O2 (mmol / m3).                   !
!                                                                      !
!=======================================================================
!  Algorithm based on Benson and Krause(1984)                          !
!=======================================================================
!
      USE mod_kinds
!
!  Function result
!
      real(r8) :: O2_sat_BK
!
!  Imported variable declarations.
!
      real(r8), intent(in) :: T
      real(r8), intent(in) :: S
!
!  Local variable declarations.
!

      real(r8), parameter :: A0 = 5.80871 
      real(r8), parameter :: A1 = 3.20291
      real(r8), parameter :: A2 = 4.17887
      real(r8), parameter :: A3 = 5.10006
      real(r8), parameter :: A4 = -9.86643*10**(-2)
      real(r8), parameter :: A5 = 3.80369
      real(r8), parameter :: B0 = -7.01577*10**(-3)
      real(r8), parameter :: B1 = -7.70028*10**(-3)
      real(r8), parameter :: B2 = -1.13864*10**(-2) 
      real(r8), parameter :: B3 = -9.51519*10**(-3)
      real(r8), parameter :: C0 = -2.75915*10**(-7)
      real(r8) :: Ts

      Ts = LOG( (298.15-T)*(273.15+T)**(-1) )
!
!     Calculate saturation O2.
!     Unit of O2 is micro-mol/kg (which is roughly equivalent to milli-mol/m3)
!
      O2_sat_BK = EXP( A0 + A1*Ts + A2*Ts**2 + A3*Ts**2 + A3*Ts**3 +        &
     &              A4*Ts**4 + A5*Ts**5 +                                &
     &              S*(B0 + B1*Ts + B2*Ts**2 + B3*Ts**3) +               &
     &              C0*S**2 )

      RETURN
      END FUNCTION O2_sat_BK



