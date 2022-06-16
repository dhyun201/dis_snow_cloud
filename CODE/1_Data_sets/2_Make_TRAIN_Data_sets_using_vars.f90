PROGRAM extract_snow

IMPLICIT NONE

INTEGER,PARAMETER                               :: sp=SELECTED_REAL_KIND(8,12)
INTEGER,PARAMETER                               :: wp=sp
INTEGER                                         :: i,j,k,ii,jj,kk, &
                                                   iday, imonth, iyear, &
                                                   jday
INTEGER(KIND=1)                                 :: mt, istat
INTEGER(KIND=2)                                 :: ct
INTEGER(KIND=4)                                 :: snow_ct, &
                                                   no_snow_ct, &
                                                   cld_ct
INTEGER(KIND=4),PARAMETER                       :: gx = 5500, gy = 5500
INTEGER(KIND=1),DIMENSION(gx,gy)                :: viirs_data, &
                                                   gk2a_cld_2km, &
                                                   gk2a_sc_2km, &
                                                   gk2a_background_sc_2km
INTEGER(KIND=2),DIMENSION(gx,gy)                :: gk2a_ch01_2km_dn, &
                                                   gk2a_ch02_2km_dn, &
                                                   gk2a_ch03_2km_dn, &
                                                   gk2a_ch04_2km_dn, &
                                                   gk2a_ch05_2km_dn, &
                                                   gk2a_ch06_2km_dn, &
                                                   gk2a_ch07_2km_dn, &
                                                   gk2a_ch14_2km_dn, &
                                                   gk2a_ch15_2km_dn, &
                                                   gk2a_sol_zen_2km_dn, &
                                                   gk2a_dem_2km, &
                                                   gk2a_lsmask_2km, &
                                                   gk2a_landcover_2km

REAL(KIND=4)                                    :: rhour, rminute
REAL(KIND=8)                                    :: GMST, RA, DEC, SDEC, CDEC
REAL(KIND=4),DIMENSION(24)                      :: out_flag
REAL(KIND=4),DIMENSION(gx,gy)                   :: gk2a_lat_2km, gk2a_lon_2km, &
                                                   gk2a_sol_zen_2km, &
                                                   gk2a_sol_azi_2km, &
                                                   gk2a_sat_zen_2km
REAL(KIND=4),DIMENSION(gx,gy)                   :: gk2a_ch01_2km_ref, &
                                                   gk2a_ch02_2km_ref, &
                                                   gk2a_ch03_2km_ref, &
                                                   gk2a_ch04_2km_ref, &
                                                   gk2a_ch05_2km_ref, &
                                                   gk2a_ch06_2km_ref, &
                                                   gk2a_ch07_2km_bt, &
                                                   gk2a_ch14_2km_bt, &
                                                   gk2a_ch15_2km_bt
REAL(KIND=wp),ALLOCATABLE,DIMENSION(:)           :: sc_ami_date_val, &
                                                   sc_viirs_date_val
REAL(KIND=wp)                                   :: sc_ami_date_val_tmp, &
                                                   sc_viirs_date_val_tmp
CHARACTER                                       :: ami_hh*2, ami_mn*2, ami_time*4, &
                                                   ami_dd*2, ami_yy*4, ami_mm*2
CHARACTER(LEN=256)                              :: tmp_str1
CHARACTER(LEN=6), ALLOCATABLE, DIMENSION(:)     :: month_loop
CHARACTER(LEN=12), ALLOCATABLE, DIMENSION(:)    :: sc_ami_date, sc_viirs_date
CHARACTER(LEN=256)                              :: ipath_viirs_filelist, &
                                                   ipath_viirs_filelist_train, &
                                                   ipath_viirs_sc, &
                                                   ipath_gk2a_cld, &
                                                   ipath_gk2a_sc, &
                                                   ipath_gk2a_l1b, &
                                                   ipath_gk2a_lsmask, &
                                                   ipath_gk2a_dem, &
                                                   ipath_gk2a_latlon, &
                                                   ipath_gk2a_landcover, &
                                                   ipath_gk2a_vza, &
                                                   ipath_gk2a_background_sc, &
                                                   opath_gk2a_sza, &
                                                   opath_gk2a_info

!!===============================================================================

REAL(KIND=4),DIMENSION(gx,gy)                   :: ndsi, ndwi, ndvi, btd_14_07, &
                                                   g_gk2a_scsi_thres
REAL(KIND=4)                                    :: ref_mean, &
                                                   ref_std, &
                                                   ref_ano, &
                                                   ref_diff(6), &
                                                   ref_diff2(6), &
                                                   ref_profile(6), &
                                                   nor_btd_14_07, &
                                                   nml_fix_low_btd = -80., &
                                                   nml_fix_high_btd = 20.


!! Setting path
ipath_viirs_filelist = '/ecoface/DHJIN/DATA_VIIRS.SC/'
ipath_viirs_filelist_train = '/data/DHJIN/PhD/Data_TrainTest/Filelist/'
ipath_gk2a_latlon    = '/data/GK2A/latlon/'
ipath_gk2a_lsmask    = '/data/GK2A/landsea/'
ipath_gk2a_dem       = '/data/GK2A/dem/'
ipath_gk2a_landcover = '/data/GK2A/landcover/'
ipath_gk2a_vza       = '/data/GK2A/vza/'
ipath_gk2a_background_sc = '/data/GK2A/background_sc/'

opath_gk2a_info      = '/data/DHJIN/PhD/VIIRS.SC_GK2A_DATA/'

!! GK-2A Lat/Lon
OPEN(31,FILE=TRIM(ipath_gk2a_latlon)//'Lat_2km.bin', &
        STATUS='OLD', ACCESS='DIRECT', RECL=gx*gy*4)
READ(31,REC=1) gk2a_lat_2km
CLOSE(31)

OPEN(31,FILE=TRIM(ipath_gk2a_latlon)//'Lon_2km.bin', &
        STATUS='OLD', ACCESS='DIRECT', RECL=gx*gy*4)
READ(31,REC=1) gk2a_lon_2km
CLOSE(31)

!! GK-2A Land/Sea Mask
OPEN(31,FILE=TRIM(ipath_gk2a_lsmask)//&
             'lsmask_2km.bin', &
        STATUS='OLD', ACCESS='DIRECT', RECL=gx*gy*2)
READ(31,REC=1) gk2a_lsmask_2km
CLOSE(31)

!! GK-2A DEM
OPEN(31,FILE=TRIM(ipath_gk2a_dem)//&
             'world_SRTM_dem_ami_2km.bin', &
        STATUS='OLD', ACCESS='DIRECT', RECL=gx*gy*2)
READ(31,REC=1) gk2a_dem_2km
CLOSE(31)

!! GK-2A Land Cover
OPEN(31,FILE=TRIM(ipath_gk2a_landcover)//&
             '2018_landcover_world_AMI_2km.bin', &
        STATUS='OLD', ACCESS='DIRECT', RECL=gx*gy*2)
READ(31,REC=1) gk2a_landcover_2km
CLOSE(31)

!! GK-2A Satellite Zenith Angle
OPEN(31,FILE=TRIM(ipath_gk2a_vza)//&
             'gk2a_ami_geo_vza_fd020ge.bin', &
        STATUS='OLD', ACCESS='DIRECT', RECL=gx*gy*4)
READ(31,REC=1) gk2a_sat_zen_2km
CLOSE(31)

!! GK-2A Background Snow Cover 
OPEN(31,FILE=TRIM(ipath_gk2a_background_sc)//&
             'Background_AMI_SC_Winter_2km_extended_25by25.bin', &
        STATUS='OLD', ACCESS='DIRECT', RECL=gx*gy*1)
READ(31,REC=1) gk2a_background_sc_2km
CLOSE(31)

PRINT*, ' [ Reading Compete : GK-2A Lat./Lon./lsmask/DEM/Lancover/VZA ] '

PRINT*, ' [ Start Code ] '

snow_ct = 0
no_snow_ct = 0
cld_ct = 0

!!-----------------------------------------------------------------
!OPEN(99,FILE=TRIM(opath_gk2a_info)//'VIIRS.SC.Product_SC.CLD.Flag_used_TRAIN.dataset_v1_mod10.bin', &
OPEN(99,FILE=TRIM(opath_gk2a_info)//'VIIRS.SC.Product_SC.CLD.Flag_NMSC.SC.Product_SC.Low.CLD.CLR_used_TRAIN.dataset_v1_mod10.bin', &
        ACCESS='DIRECT', RECL=24*4)

!OPEN(98,FILE=TRIM(opath_gk2a_info)//'TRAIN.dataset_v1_mod10.txt')
!!-----------------------------------------------------------------

OPEN(11,FILE=TRIM(ipath_viirs_filelist_train)//'date_sc_filelist_train.test_TRAIN_dataset.txt',&
        STATUS='OLD')
ct = 0
DO WHILE(.TRUE.)
  READ(11,*,IOSTAT=istat) tmp_str1
  IF ( istat /= 0 ) THEN
    PRINT*, ' [ Counting Complete ] ', ct
    EXIT
  ENDIF
  ct = ct + 1 
ENDDO
REWIND(11)
 
ALLOCATE(sc_ami_date(ct), sc_viirs_date(ct))
ALLOCATE(sc_ami_date_val(ct), sc_viirs_date_val(ct))

ct = ct - 1
READ(11,*) tmp_str1

!! Loop date
DO kk = 1, ct

  READ(11,*) sc_ami_date(kk), sc_viirs_date(kk)

  READ(sc_ami_date(kk), '(f13.0)') sc_ami_date_val_tmp
  READ(sc_viirs_date(kk), '(f13.0)') sc_viirs_date_val_tmp


!!-------------------------------------------------------------
!!-------------------------------------------------------------
!!-------------------------------------------------------------
  IF ( MOD(kk, 10) /= 0 ) CYCLE
!!-------------------------------------------------------------
!!-------------------------------------------------------------
!!-------------------------------------------------------------

  !! Setting day & time
  ami_yy   = sc_ami_date(kk)(1:4)
  ami_mm   = sc_ami_date(kk)(5:6)
  ami_dd   = sc_ami_date(kk)(7:8)
  ami_time = sc_ami_date(kk)(9:12)
  ami_hh   = ami_time(1:2)
  ami_mn   = ami_time(3:4)

  !! Setting path 2
  ipath_viirs_sc = '/ecoface/DHJIN/DATA_VIIRS.SC/2_AMI.File&
                    /VIIRS_Snow_AMI_Map_'//TRIM(ami_yy)//TRIM(ami_mm)//'/'                          

  PRINT*, ' Step 1 [ Setting GK-2A Time : '//sc_ami_date(kk)//' ] '

  !! Setting path 3
  ipath_gk2a_cld = '/data/DHJIN/DATA_GK2A.AMI/CLD/1_BIN.File/'//&
                   TRIM(ami_yy)//TRIM(ami_mm)//'/'//&
                   TRIM(ami_dd)//'/'//TRIM(ami_hh)//'/'
  ipath_gk2a_sc  = '/ecoface2/DHJIN/DATA_GK2A.SCSI/SCSI_bin/'//&
                   TRIM(ami_yy)//TRIM(ami_mm)//'/'
  ipath_gk2a_l1b = '/data/DHJIN/DATA_GK2A.AMI/L1B/1_BIN.File/'//&
                   TRIM(ami_yy)//TRIM(ami_mm)//'/'//&
                   TRIM(ami_dd)//'/'//TRIM(ami_hh)//'/'
  opath_gk2a_sza = '/data/DHJIN/DATA_GK2A.AMI/SZA/'//&
                   TRIM(ami_yy)//TRIM(ami_mm)//'/'//&
                   TRIM(ami_dd)//'/'//TRIM(ami_hh)//'/'
  CALL system('mkdir -p '//TRIM(opath_gk2a_sza))

    !! VIIRS Snow Cover AMI Map 
    OPEN(31,FILE=TRIM(ipath_viirs_sc)//'VIIRS.SC_AMI_Map_2km_'//&
                 TRIM(sc_viirs_date(kk))//'.bin', &
            STATUS='OLD', ACCESS='DIRECT', RECL=gx*gy*1, IOSTAT=istat)
    IF ( istat /= 0 ) CYCLE
    READ(31,REC=1) viirs_data
    CLOSE(31)
    PRINT*, ' Step 2 [ Reading Complete : VIIRS Snow Cover ] '

    !! GK-2A AMI Dataset
    !! Snow Cover
    OPEN(21,FILE=TRIM(ipath_gk2a_sc)//'gk2a_ami_le2_scsi_fd020ge_'//&
                 sc_ami_date(kk)//'.bin', &
            STATUS='OLD', ACCESS='DIRECT', RECL=gx*gy*1,IOSTAT=istat)
    IF ( istat /= 0 ) CYCLE
    READ(21,REC=1) gk2a_sc_2km
    CLOSE(21)
    PRINT*, ' Step 3 [ Reading Complete : GK-2A Snow Cover ] '

    !! Cloud Mask
    OPEN(21,FILE=TRIM(ipath_gk2a_cld)//'gk2a_ami_le2_cld_fd020ge_'//&
                 sc_ami_date(kk)//'.bin', &
            STATUS='OLD', ACCESS='DIRECT', RECL=gx*gy*1,IOSTAT=istat)
    IF ( istat /= 0 ) CYCLE
    READ(21,REC=1) gk2a_cld_2km
    CLOSE(21)
    PRINT*, ' Step 3 [ Reading Complete : GK-2A Cloud Mask ] '

    !! Channel Data 01
    OPEN(21,FILE=TRIM(ipath_gk2a_l1b)//&
                 'gk2a_ami_le1b_vi004_fd020ge_'//TRIM(sc_ami_date(kk))//'.bin', &
            STATUS='OLD', ACCESS='DIRECT', RECL=gx*gy*2, IOSTAT=istat)
    IF ( istat /= 0 ) CYCLE
    READ(21, REC=1) gk2a_ch01_2km_dn
    CLOSE(21)

    !! Channel Data 02
    OPEN(21,FILE=TRIM(ipath_gk2a_l1b)//&
                 'gk2a_ami_le1b_vi005_fd020ge_'//TRIM(sc_ami_date(kk))//'.bin', &
            STATUS='OLD', ACCESS='DIRECT', RECL=gx*gy*2, IOSTAT=istat)
    IF ( istat /= 0 ) CYCLE
    READ(21, REC=1) gk2a_ch02_2km_dn
    CLOSE(21)

    !! Channel Data 03
    OPEN(21,FILE=TRIM(ipath_gk2a_l1b)//&
                 'gk2a_ami_le1b_vi006_fd020ge_'//TRIM(sc_ami_date(kk))//'.bin', &
            STATUS='OLD', ACCESS='DIRECT', RECL=gx*gy*2, IOSTAT=istat)
    IF ( istat /= 0 ) CYCLE
    READ(21, REC=1) gk2a_ch03_2km_dn
    CLOSE(21)

    !! Channel Data 04
    OPEN(21,FILE=TRIM(ipath_gk2a_l1b)//&
                 'gk2a_ami_le1b_vi008_fd020ge_'//TRIM(sc_ami_date(kk))//'.bin', &
            STATUS='OLD', ACCESS='DIRECT', RECL=gx*gy*2, IOSTAT=istat)
    IF ( istat /= 0 ) CYCLE
    READ(21, REC=1) gk2a_ch04_2km_dn
    CLOSE(21)

    !! Channel Data 05
    OPEN(21,FILE=TRIM(ipath_gk2a_l1b)//&
                 'gk2a_ami_le1b_nr013_fd020ge_'//TRIM(sc_ami_date(kk))//'.bin', &
            STATUS='OLD', ACCESS='DIRECT', RECL=gx*gy*2, IOSTAT=istat)
    IF ( istat /= 0 ) CYCLE
    READ(21, REC=1) gk2a_ch05_2km_dn
    CLOSE(21)

    !! Channel Data 06
    OPEN(21,FILE=TRIM(ipath_gk2a_l1b)//&
                 'gk2a_ami_le1b_nr016_fd020ge_'//TRIM(sc_ami_date(kk))//'.bin', &
            STATUS='OLD', ACCESS='DIRECT', RECL=gx*gy*2, IOSTAT=istat)
    IF ( istat /= 0 ) CYCLE
    READ(21, REC=1) gk2a_ch06_2km_dn
    CLOSE(21)

    !! Channel Data 07
    OPEN(21,FILE=TRIM(ipath_gk2a_l1b)//&
                 'gk2a_ami_le1b_sw038_fd020ge_'//TRIM(sc_ami_date(kk))//'.bin', &
            STATUS='OLD', ACCESS='DIRECT', RECL=gx*gy*2, IOSTAT=istat)
    IF ( istat /= 0 ) CYCLE
    READ(21, REC=1) gk2a_ch07_2km_dn
    CLOSE(21)

    !! Channel Data 14
    OPEN(21,FILE=TRIM(ipath_gk2a_l1b)//&
                 'gk2a_ami_le1b_ir112_fd020ge_'//TRIM(sc_ami_date(kk))//'.bin', &
            STATUS='OLD', ACCESS='DIRECT', RECL=gx*gy*2, IOSTAT=istat)
    IF ( istat /= 0 ) CYCLE
    READ(21, REC=1) gk2a_ch14_2km_dn
    CLOSE(21)

    !! Channel Data 15
    OPEN(21,FILE=TRIM(ipath_gk2a_l1b)//&
                 'gk2a_ami_le1b_ir123_fd020ge_'//TRIM(sc_ami_date(kk))//'.bin', &
            STATUS='OLD', ACCESS='DIRECT', RECL=gx*gy*2, IOSTAT=istat)
    IF ( istat /= 0 ) CYCLE
    READ(21, REC=1) gk2a_ch15_2km_dn
    CLOSE(21)

  !! Convert to Ref. and BT
  gk2a_ch01_2km_ref = gk2a_ch01_2km_dn * 0.0001
  gk2a_ch02_2km_ref = gk2a_ch02_2km_dn * 0.0001
  gk2a_ch03_2km_ref = gk2a_ch03_2km_dn * 0.0001
  gk2a_ch04_2km_ref = gk2a_ch04_2km_dn * 0.0001
  gk2a_ch05_2km_ref = gk2a_ch05_2km_dn * 0.0001
  gk2a_ch06_2km_ref = gk2a_ch06_2km_dn * 0.0001

  gk2a_ch07_2km_bt  = (gk2a_ch07_2km_dn/100.) + 100.
  gk2a_ch14_2km_bt  = (gk2a_ch14_2km_dn/100.) + 100.
  gk2a_ch15_2km_bt  = (gk2a_ch15_2km_dn/100.) + 100.

  PRINT*, ' Step 4 [ Reading & Coverting Complete : GK-2A Channels & Cloud Mask ] '

  !! Solar Zenith Angle
  READ(ami_yy, '(i4)') iyear
  READ(ami_mm, '(i2)') imonth
  READ(ami_dd, '(i2)') iday
  READ(ami_hh, *) rhour
  READ(ami_mn, *) rminute

  CALL JULIAN(iday, imonth, iyear, jday)
  rhour = rhour + rminute/60.
  CALL AUX_Calc_GMST_RA_DEC(iyear, jday, rhour, GMST, RA, DEC)
  SDEC = SIN(DEC) ; CDEC = COS(DEC)

!!-----------------------------------------------------------------------
  !! Loop Pixel 
  DO j = 1, gy
    IF (MAXVAL(gk2a_lsmask_2km(:,j)) == -999) THEN
      gk2a_sol_zen_2km_dn(:,j) = -999
      CYCLE
    ENDIF

    DO i = 1, gx
        !! Check fill-value & sea
        IF (gk2a_lsmask_2km(i,j) == -999) THEN
          CYCLE
        ELSE IF (gk2a_lsmask_2km(i,j) == 0) THEN
          CYCLE
        ENDIF

        !! Check Background Snow Cover
        IF (gk2a_background_sc_2km(i,j) /= 1) CYCLE

        !! Check viirs data
        IF ( viirs_data(i,j) == -128 ) CYCLE
        IF ( viirs_data(i,j) ==  -10 ) CYCLE
        IF ( viirs_data(i,j) ==   10 ) CYCLE

        !! Calc Solar Angles
        CALL AUX_Calc_Sol_Zenith_Azimuth_Angle(GMST, RA, SDEC, CDEC, &
                                               gk2a_lon_2km(i,j), &
                                               gk2a_lat_2km(i,j), &
                                               gk2a_sol_zen_2km(i,j), &
                                               gk2a_sol_azi_2km(i,j))             

        !! Night Check (SZA > 80)
        IF ( gk2a_sol_zen_2km(i,j) > 80.0 ) CYCLE

        !! High Cloudy Flag Check (GK-2A CLD)
        IF ( gk2a_cld_2km(i,j) == 0 ) CYCLE

        !! Min check
        IF ( gk2a_ch01_2km_ref(i,j) < 0.001 ) gk2a_ch01_2km_ref(i,j) = 0.001                                          
        IF ( gk2a_ch02_2km_ref(i,j) < 0.001 ) gk2a_ch02_2km_ref(i,j) = 0.001                                          
        IF ( gk2a_ch03_2km_ref(i,j) < 0.001 ) gk2a_ch03_2km_ref(i,j) = 0.001                                          
        IF ( gk2a_ch04_2km_ref(i,j) < 0.001 ) gk2a_ch04_2km_ref(i,j) = 0.001                                          
        IF ( gk2a_ch05_2km_ref(i,j) < 0.001 ) gk2a_ch05_2km_ref(i,j) = 0.001                                          
        IF ( gk2a_ch06_2km_ref(i,j) < 0.001 ) gk2a_ch06_2km_ref(i,j) = 0.001                                          

        !! Normalizing refelctance using sza
        gk2a_ch01_2km_ref(i,j) = gk2a_ch01_2km_ref(i,j) / COSD(gk2a_sol_zen_2km(i,j))
        IF ( gk2a_ch01_2km_ref(i,j) > 1.0 ) gk2a_ch01_2km_ref(i,j) = 1.0

        gk2a_ch02_2km_ref(i,j) = gk2a_ch02_2km_ref(i,j) / COSD(gk2a_sol_zen_2km(i,j))
        IF ( gk2a_ch02_2km_ref(i,j) > 1.0 ) gk2a_ch02_2km_ref(i,j) = 1.0

        gk2a_ch03_2km_ref(i,j) = gk2a_ch03_2km_ref(i,j) / COSD(gk2a_sol_zen_2km(i,j))
        IF ( gk2a_ch03_2km_ref(i,j) > 1.0 ) gk2a_ch03_2km_ref(i,j) = 1.0

        gk2a_ch04_2km_ref(i,j) = gk2a_ch04_2km_ref(i,j) / COSD(gk2a_sol_zen_2km(i,j))
        IF ( gk2a_ch04_2km_ref(i,j) > 1.0 ) gk2a_ch04_2km_ref(i,j) = 1.0

        gk2a_ch05_2km_ref(i,j) = gk2a_ch05_2km_ref(i,j) / COSD(gk2a_sol_zen_2km(i,j))
        IF ( gk2a_ch05_2km_ref(i,j) > 1.0 ) gk2a_ch05_2km_ref(i,j) = 1.0

        gk2a_ch06_2km_ref(i,j) = gk2a_ch06_2km_ref(i,j) / COSD(gk2a_sol_zen_2km(i,j))
        IF ( gk2a_ch06_2km_ref(i,j) > 1.0 ) gk2a_ch06_2km_ref(i,j) = 1.0

        !! Calc NDSI, NDVI, NDWI, BTD[11.2-3.8], Nor. BTD[11.2-3.8], 1.61 Anomaly
        ndsi(i,j) = ( gk2a_ch03_2km_ref(i,j) - gk2a_ch06_2km_ref(i,j) ) / &
                    ( gk2a_ch03_2km_ref(i,j) + gk2a_ch06_2km_ref(i,j) ) 
        ndwi(i,j) = ( gk2a_ch04_2km_ref(i,j) - gk2a_ch06_2km_ref(i,j) ) / &
                    ( gk2a_ch04_2km_ref(i,j) + gk2a_ch06_2km_ref(i,j) )  
        ndvi(i,j) = ( gk2a_ch04_2km_ref(i,j) - gk2a_ch03_2km_ref(i,j) ) / &
                    ( gk2a_ch04_2km_ref(i,j) + gk2a_ch03_2km_ref(i,j) ) 
        btd_14_07(i,j) = gk2a_ch14_2km_bt(i,j) - gk2a_ch07_2km_bt(i,j)

        IF (btd_14_07(i,j) < nml_fix_low_btd) btd_14_07(i,j) = nml_fix_low_btd
        IF (btd_14_07(i,j) > nml_fix_high_btd) btd_14_07(i,j) = nml_fix_high_btd
        nor_btd_14_07 = 0.
        nor_btd_14_07 = ( btd_14_07(i,j) - nml_fix_low_btd ) / &
                             ( nml_fix_high_btd - nml_fix_low_btd )  

        ref_profile(1) = gk2a_ch01_2km_ref(i,j)
        ref_profile(2) = gk2a_ch02_2km_ref(i,j)
        ref_profile(3) = gk2a_ch03_2km_ref(i,j)
        ref_profile(4) = gk2a_ch04_2km_ref(i,j)
        ref_profile(5) = gk2a_ch05_2km_ref(i,j)
        ref_profile(6) = gk2a_ch06_2km_ref(i,j)
        ref_mean = SUM(ref_profile(:)) / 6.
        ref_diff = ref_profile(:) - ref_mean
        ref_diff2 = ref_diff * ref_diff
        ref_std = SQRT(SUM(ref_diff2) / 5.)
        ref_ano = (ref_profile(6) - ref_mean) / ref_std
   
        g_gk2a_scsi_thres(i,j) = ( gk2a_ch06_2km_ref(i,j) / gk2a_ch02_2km_ref(i,j) ) * &
                                 ( gk2a_ch14_2km_bt(i,j) - gk2a_ch07_2km_bt(i,j) )
 
        !! Input the information for out_flag
        out_flag(1) = gk2a_ch01_2km_ref(i,j)    
        out_flag(2) = gk2a_ch02_2km_ref(i,j)    
        out_flag(3) = gk2a_ch03_2km_ref(i,j)    
        out_flag(4) = gk2a_ch04_2km_ref(i,j)    
        out_flag(5) = gk2a_ch05_2km_ref(i,j)    
        out_flag(6) = gk2a_ch06_2km_ref(i,j)    
        out_flag(7) = gk2a_ch07_2km_bt(i,j)    
        out_flag(8) = gk2a_ch14_2km_bt(i,j)    
        out_flag(9) = gk2a_ch15_2km_bt(i,j)    
  
        out_flag(10) = gk2a_sol_zen_2km(i,j)
        out_flag(11) = gk2a_sat_zen_2km(i,j)    
        out_flag(12) = gk2a_lat_2km(i,j)
        out_flag(13) = gk2a_lon_2km(i,j)        
        out_flag(14) = gk2a_cld_2km(i,j)
        out_flag(15) = gk2a_sc_2km(i,j)
        out_flag(16) = viirs_data(i,j)
        out_flag(17) = gk2a_dem_2km(i,j)
        out_flag(18) = gk2a_landcover_2km(i,j)

        out_flag(19) = ndsi(i,j)
        out_flag(20) = ndvi(i,j)
        out_flag(21) = ndwi(i,j)
        out_flag(22) = btd_14_07(i,j)
        out_flag(23) = nor_btd_14_07 
        out_flag(24) = ref_ano 

        snow_ct = snow_ct + 1
        WRITE(99, REC=snow_ct) out_flag

        !! Write Snow/No Snow/Cloud File
!        IF ( viirs_data(i,j) == 1 ) THEN   !! Snow
!          snow_ct = snow_ct + 1
!          WRITE(99, REC=snow_ct) out_flag
!        ELSE IF ( viirs_data(i,j) == 3 ) THEN   !! Cloud
!          snow_ct = snow_ct + 1
!          WRITE(99, REC=snow_ct) out_flag
!        ENDIF

      ENDDO  !! X Loop END
    ENDDO  !! Y Loop END

!!-----------------------------------------------------------------------

  PRINT*, ' Step 5 [ Inputting & Writing Complete ] '//sc_ami_date(kk)
  PRINT*, ' ---------------------------------------------------------- '

ENDDO  !! Scene Loop END
 
DEALLOCATE(sc_ami_date, sc_viirs_date)

CLOSE(11) 
CLOSE(99)

PRINT*, ' Snow & Cloud Count : ', snow_ct

END PROGRAM

!!===========================================================
!!===========================================================

SUBROUTINE JULIAN(iday,imonth,iyear,jday)

!arguments
INTEGER, INTENT(IN) ::  iday, imonth, iyear
INTEGER, INTENT(OUT) :: jday

!local variables
INTEGER ::  j
INTEGER, DIMENSION(12) :: jmonth = (/31,28,31,30,31,30,31,31,30,31,30,31/)

jday = iday
IF((MOD(iyear,4)==0).AND..NOT.(MOD(iyear,100)==0).OR.(MOD(iyear,400)==0)) THEN 
   jmonth(2)=29
ENDIF

DO j = 1, imonth - 1
  jday = jday + jmonth(j)
END DO

END SUBROUTINE JULIAN

!!===========================================================
!!===========================================================


SUBROUTINE AUX_Calc_GMST_RA_DEC(year, day, hour,GMST,RA,DEC)

IMPLICIT NONE

INTEGER, INTENT(IN)  :: YEAR, DAY
REAL,    INTENT(IN)  :: HOUR

REAL(8),    INTENT(OUT) :: GMST, RA, DEC

REAL, PARAMETER      :: pi = 3.1415926535898, &
                        pi2 = pi*2., &
                        deg2rad = pi/180.
                     
INTEGER   DELTA, LEAP
DOUBLE PRECISION  DEN, ECLONG, HA, JD, LMST,   &
                  MNANOM, MNLONG, NUM, OBLQEC, &
                  REFRAC, TIME

!    ** current Julian date (actually add 2,400,000
!    ** for true JD);  LEAP = leap days since 1949;
!    ** 32916.5 is midnite 0 jan 1949 minus 2.4e6

DELTA  = YEAR - 1949
LEAP   = DELTA / 4
JD     = 32916.5 + (DELTA*365 + LEAP + DAY) + HOUR / 24.

!                    ** last yr of century not leap yr unless divisible
!                    ** by 400 (not executed for the allowed YEAR range,
!                    ** but left in so our successors can adapt this for
!                    ** the following 100 years)

IF( MOD( YEAR, 100 ).EQ.0 .AND.        &
    MOD( YEAR, 400 ).NE.0 ) JD = JD - 1.

!                     ** ecliptic coordinates
!                     ** 51545.0 + 2.4e6 = noon 1 jan 2000

TIME  = JD - 51545.0

!                    ** force mean longitude between 0 and 360 degs
MNLONG = 280.460 + 0.9856474*TIME
MNLONG = MOD( MNLONG, 360.D0 )
IF( MNLONG.LT.0. ) MNLONG = MNLONG + 360.

!                    ** mean anomaly in radians between 0 and 2*pi

MNANOM = 357.528 + 0.9856003*TIME
MNANOM = MOD( MNANOM, 360.D0 )
IF( MNANOM.LT.0.) MNANOM = MNANOM + 360.

MNANOM = MNANOM*deg2rad

!                    ** ecliptic longitude and obliquity
!                    ** of ecliptic in radians

ECLONG = MNLONG + 1.915*SIN( MNANOM ) + 0.020*SIN( 2.*MNANOM )
ECLONG = MOD( ECLONG, 360.D0 )
IF( ECLONG.LT.0. ) ECLONG = ECLONG + 360.

OBLQEC = 23.439 - 0.0000004*TIME
ECLONG = ECLONG*deg2rad
OBLQEC = OBLQEC*deg2rad

!                    ** right ascension

NUM  = COS( OBLQEC )*SIN( ECLONG )
DEN  = COS( ECLONG )
RA   = ATAN( NUM / DEN )

!                    ** Force right ascension between 0 and 2*pi

IF( DEN.LT.0.0 ) THEN
  RA  = RA + pi
ELSE IF( NUM.LT.0.0 ) THEN
  RA  = RA + pi2
END IF

!                    ** declination

DEC  = ASIN( SIN( OBLQEC )*SIN( ECLONG ) )

!                    ** Greenwich mean sidereal time in hours
GMST = 6.697375 + 0.0657098242*TIME + HOUR

!                    ** Hour not changed to sidereal time since
!                    ** 'time' includes the fractional day

GMST  = MOD( GMST, 24.D0)
IF( GMST.LT.0. ) GMST   = GMST + 24.

END SUBROUTINE AUX_Calc_GMST_RA_DEC


!!===========================================================
!!===========================================================

SUBROUTINE AUX_Calc_Sol_Zenith_Azimuth_Angle(GMST,RA,SDEC,CDEC, lon, lat, zen, azi)

      IMPLICIT NONE

!      INTEGER, INTENT(IN)  :: YEAR, DAY
!      REAL,    INTENT(IN)  :: HOUR, LAT, LON
      REAL,    INTENT(IN)  :: LAT, LON
      REAL(8),    INTENT(IN)  :: GMST,RA, SDEC,CDEC

      REAL,    INTENT(OUT) :: ZEN, AZI

!     .. Local Scalars ..
      REAL                 :: EL

      REAL                 :: SLAT, CLAT

      REAL, PARAMETER      :: pi = 3.1415926535898, &
                             pi2 = pi*2., &
                           deg2rad = pi/180.


      INTEGER   DELTA, LEAP
      DOUBLE PRECISION  DEN, ECLONG, HA, JD, LMST,   &
                        MNANOM, MNLONG, NUM, OBLQEC, &
                        REFRAC, TIME
      REAL(8) r8tmp

!     ..
!     .. Intrinsic Functions ..

      INTRINSIC AINT, ASIN, ATAN, COS, MOD, SIN, TAN
!     ..

!                    ** current Julian date (actually add 2,400,000
!                    ** for true JD);  LEAP = leap days since 1949;
!                    ** 32916.5 is midnite 0 jan 1949 minus 2.4e6


!                    ** local mean sidereal time in radians
      LMST  = GMST + LON / 15.
      LMST  = MOD( LMST, 24.D0 )
      IF( LMST.LT.0. ) LMST   = LMST + 24.

      LMST   = LMST*15.*deg2rad

!                    ** hour angle in radians between -pi and pi

      HA  = LMST - RA

      IF( HA .LT. -pi ) HA  = HA + pi2
      IF( HA .GT. pi )   HA  = HA - pi2

!                    ** solar azimuth and elevation
      SLAT = SIN(LAT*deg2rad)
      CLAT = COS(LAT*deg2rad)

      r8tmp =  SDEC*SLAT + CDEC*CLAT*COS(HA)
      IF(r8tmp > 1.0) r8tmp = 1.0
      EL = ASIN(r8tmp)


      AZI  = - CDEC*SIN( HA ) / COS( EL )

! Check for the round off error 2017. 07. 10.
      AZI  = max( -1.0, min( AZI, 1.0 ) )
      AZI  = ASIN( AZI )

!                    ** Put azimuth between 0 and 2*pi radians

!      IF( SIN( DEC ) - SIN( EL )*SIN( LAT*deg2rad ) .GE. 0. ) THEN
      IF( SDEC - SIN( EL )*SLAT .GE. 0. ) THEN
         IF( SIN(AZI).LT.0.) AZI  = AZI + pi2
      ELSE
         AZI  = pi - AZI
      END IF

!                     ** Convert elevation and azimuth to degrees
      EL  = EL / deg2rad
      AZI  = AZI / deg2rad

!                     ** Convert azimuth to -180 to 180 ranges
      IF ( AZI .GT. 180. ) AZI = AZI - 360.

!!  ======== Refraction correction for U.S. Standard Atmos. ==========
!!      (assumes elevation in degs) (3.51823=1013.25 mb/288 K)
!
!      IF( EL.GE.19.225 ) THEN
!
!         REFRAC = 0.00452*3.51823 / TAN( EL*deg2rad )
!
!      ELSE IF( EL.GT.-0.766 .AND. EL.LT.19.225 ) THEN
!
!         REFRAC = 3.51823 * ( 0.1594 + EL*(0.0196 + 0.00002*EL) ) /  &
!                  ( 1. + EL*(0.505 + 0.0845*EL) )
!
!      ELSE IF( EL.LE.-0.766 ) THEN
!
         REFRAC = 0.0
!
!      END IF

! sm: switch off refraction:

      EL  = EL + REFRAC

      ZEN = 90. - EL

   END SUBROUTINE AUX_Calc_Sol_Zenith_Azimuth_Angle
   !===============================================================================!
   !===============================================================================!
