PROGRAM geographic_latlon

IMPLICIT NONE

INTEGER,PARAMETER                               :: sp=SELECTED_REAL_KIND(6,37)
INTEGER,PARAMETER                               :: wp=sp
INTEGER(KIND=1)                                 :: istat
INTEGER(KIND=2)                                 :: i, j, k, ct, mt, kk
INTEGER(KIND=4)                                 :: jul_int, year_int, vx, vy
INTEGER(KIND=4),PARAMETER                       :: wx=19489, wy=19480, &  !! 1km pixel size
                                                   gx=5500, gy=5500
INTEGER(KIND=1), ALLOCATABLE, DIMENSION(:,:)    :: viirs_data
INTEGER(KIND=2), ALLOCATABLE, DIMENSION(:,:)    :: geo_index_x_viirs, &
                                                   geo_index_y_viirs
INTEGER(KIND=1),DIMENSION(wx,wy)                :: geo_viirs_snow
INTEGER(KIND=1),DIMENSION(gx,gy)                :: snow_ami
INTEGER(KIND=2),DIMENSION(gx,gy)                :: landsea_ami, &
                                                   geo_index_x_ami, &
                                                   geo_index_y_ami
REAL(KIND=wp),PARAMETER                         :: pixel_size=0.00833333     !! 1km pixel size 
REAL(wp)                                        :: stlat=81.169579-pixel_size/2., &
                                                   stlon=46.994072+pixel_size/2.
REAL(KIND=wp),DIMENSION(wx,wy)                  :: lat_geo, lon_geo
REAL(KIND=wp),DIMENSION(gx,gy)                  :: lat_ami, lon_ami
REAL(KIND=4), ALLOCATABLE, DIMENSION(:,:)       :: lat_viirs, lon_viirs
CHARACTER                                       :: date*8, time*4, fulldate*27, month_loop_str*6, &
                                                   jul_date
CHARACTER(LEN=36)                               :: viirs_fullname
CHARACTER(LEN=256)                              :: str_tmp1
CHARACTER(LEN=6),ALLOCATABLE,DIMENSION(:)       :: month_loop
CHARACTER(LEN=256)                              :: input_ami_latlon, &
                                                   input_ami_landsea, &
                                                   input_viirs, &
                                                   output_path



!! Setting path
input_ami_latlon  = '/ecoface/DHJIN/DATA_GK-2A.AMI/latlon/'
input_ami_landsea = '/ecoface/DHJIN/DATA_GK-2A.AMI/landsea/'

!! Setting Month Loop No.
PRINT*, ' Input Month Loop No. (Integer) '
READ(*,*) mt

IF (.NOT. ALLOCATED(month_loop)) ALLOCATE(month_loop(mt))


!! Setting a Global Lat./Lon.
DO j=1,wy
   lat_geo(:,j) = stlat - pixel_size * (j-1)
   DO i=1,wx
      lon_geo(i,j) = stlon + pixel_size * (i-1)
   ENDDO
ENDDO

!! Open & Read GK-2A AMI Lat./Lon.
OPEN(31,FILE=TRIM(input_ami_latlon)//'Lat_2km.bin',ACCESS='DIRECT',RECL=gx*gy*4)
READ(31,REC=1) lat_ami
CLOSE(31)

OPEN(31,FILE=TRIM(input_ami_latlon)//'Lon_2km.bin',ACCESS='DIRECT',RECL=gx*gy*4)
READ(31,REC=1) lon_ami
CLOSE(31)

WHERE( lon_ami < 0. .and. lon_ami > -200.) lon_ami = lon_ami + 360.

OPEN(31,FILE=TRIM(input_ami_landsea)//'lsmask_2km.bin', &
        ACCESS='DIRECT', STATUS='OLD', RECL=gx*gy*2)
READ(31,REC=1) landsea_ami
CLOSE(31)

!!================================================================================
DO kk = 1, mt
  PRINT*, ' Input the month_loop(kk) / ex) 202001 '
  READ(*,*) month_loop(kk)
ENDDO

!!================================================================================
DO kk = 1, mt
!!================================================================================

!! Convert month loop (integer -> string)
READ(month_loop(kk), '(a6)') month_loop_str

!!---------------------------------------------------------------------------
!! Setting Path
input_viirs = '/ecoface/DHJIN/DATA_VIIRS.SC/1_BIN.File/'//TRIM(month_loop_str)//'/'
output_path = '/ecoface/DHJIN/DATA_VIIRS.SC/2_AMI.File/VIIRS_Snow_AMI_Map_'&
              //TRIM(month_loop_str)//'/'

CALL output_folder(output_path)
!!---------------------------------------------------------------------------

!! Open & Read S-NPP VIIRS File List
OPEN(11,FILE=TRIM(input_viirs)//'vnp10_filelist_'//month_loop(kk)//'.txt', STATUS='OLD')

ct = 0
DO WHILE(.True.)
   READ(11,*,IOSTAT=istat) str_tmp1 
   IF ( istat /= 0 ) THEN
      EXIT
   ENDIF
   ct = ct + 1
ENDDO

PRINT*, ' VIIRS Files No. : ', ct

REWIND(11)

!! VIIRS FIles Loop Start
DO k = 1, ct
   READ(11,*) viirs_fullname, vx, vy !! VNP10.A2019336.0800.001.201933717122
   
!! Setting date, time
   READ(viirs_fullname(8:11), '(i4)') year_int
   READ(viirs_fullname(12:14), '(i3)') jul_int
 
   CALL cal_date(year_int, jul_int, date)

   time     = viirs_fullname(16:19)  !! 0800
   fulldate = date//'.'//time        !! 20191202.0800

!! Allocate dimension
   IF (.NOT. ALLOCATED(viirs_data)) ALLOCATE(viirs_data(vx,vy))   
   IF (.NOT. ALLOCATED( lat_viirs)) ALLOCATE( lat_viirs(vx,vy))   
   IF (.NOT. ALLOCATED( lon_viirs)) ALLOCATE( lon_viirs(vx,vy))   
   IF (.NOT. ALLOCATED(geo_index_x_viirs)) ALLOCATE(geo_index_x_viirs(vx,vy))   
   IF (.NOT. ALLOCATED(geo_index_y_viirs)) ALLOCATE(geo_index_y_viirs(vx,vy))   

!! VIIRS Snow Cover
   OPEN(21,FILE=TRIM(input_viirs)//TRIM(viirs_fullname)//'_QA.NDSI.SC.bin', &
           STATUS='OLD', ACCESS='DIRECT', RECL=vx*vy*1, IOSTAT=istat)
     IF ( istat /= 0 ) PRINT*, ' No File : NDSI.SC '
   READ(21,REC=1) viirs_data
   CLOSE(21)

!! VIIRS Latitude
   OPEN(21,FILE=TRIM(input_viirs)//TRIM(viirs_fullname)//'_latitude.bin', &
           STATUS='OLD', ACCESS='DIRECT', RECL=vx*vy*4, IOSTAT=istat)
     IF ( istat /= 0 ) PRINT*, ' No File : Latitude '
   READ(21,REC=1) lat_viirs
   CLOSE(21)

!! VIIRS Longitude
   OPEN(21,FILE=TRIM(input_viirs)//TRIM(viirs_fullname)//'_longitude.bin', &
           STATUS='OLD', ACCESS='DIRECT', RECL=vx*vy*4, IOSTAT=istat)
     IF ( istat /= 0 ) PRINT*, ' No File : Longitude '
   READ(21,REC=1) lon_viirs
   CLOSE(21)

   WHERE(lon_viirs < 0.) lon_viirs = lon_viirs + 360.


   geo_index_x_viirs = -999
   geo_index_y_viirs = -999
   geo_viirs_snow    = -99

   DO j = 1, vy
      DO i = 1, vx
         IF (viirs_data(i,j) == -99) CYCLE  !! FILL-VALUE Check
         IF (viirs_data(i,j) == -10) CYCLE  !! Sea Check

      IF (viirs_data(i,j) > 0 .AND. viirs_data(i,j) <= 100) THEN
        geo_index_x_viirs(i,j) = NINT( (lon_viirs(i,j)-stlon)/pixel_size )+1
        geo_index_y_viirs(i,j) = NINT( (stlat-lat_viirs(i,j))/pixel_size )+1
        geo_viirs_snow(geo_index_x_viirs(i,j),geo_index_y_viirs(i,j)) = 1  !! VIIRS Snow    
      ELSE IF (viirs_data(i,j) == 150-256) THEN
        geo_index_x_viirs(i,j) = NINT( (lon_viirs(i,j)-stlon)/pixel_size )+1
        geo_index_y_viirs(i,j) = NINT( (stlat-lat_viirs(i,j))/pixel_size )+1
        geo_viirs_snow(geo_index_x_viirs(i,j),geo_index_y_viirs(i,j)) = 2  !! VIIRS Bad Snow
      ELSE IF (viirs_data(i,j) == 0) THEN
        geo_index_x_viirs(i,j) = NINT( (lon_viirs(i,j)-stlon)/pixel_size )+1
        geo_index_y_viirs(i,j) = NINT( (stlat-lat_viirs(i,j))/pixel_size )+1
        geo_viirs_snow(geo_index_x_viirs(i,j),geo_index_y_viirs(i,j)) = 0  !! VIIRS No Snow
      ELSE IF (viirs_data(i,j) == 250-256) THEN
        geo_index_x_viirs(i,j) = NINT( (lon_viirs(i,j)-stlon)/pixel_size )+1
        geo_index_y_viirs(i,j) = NINT( (stlat-lat_viirs(i,j))/pixel_size )+1
        geo_viirs_snow(geo_index_x_viirs(i,j),geo_index_y_viirs(i,j)) = 3  !! VIIRS Cloud
      ENDIF

      ENDDO
   ENDDO

   PRINT*, ' [ Convert VIIRS data to Gegraphic Map ] '

!!------------------------------------------------------------------------------
   snow_ami = -128

   WHERE(landsea_ami == 1) snow_ami =  10
   WHERE(landsea_ami == 0) snow_ami = -10
  
   DO j = 1, gy
      IF ( MAXVAL(landsea_ami(:,j)) == -999 ) CYCLE
      DO i = 1, gx
         IF ( landsea_ami(i,j) == -999 ) CYCLE

         geo_index_x_ami(i,j) = NINT( (lon_ami(i,j)-stlon)/pixel_size )+1
         geo_index_y_ami(i,j) = NINT( (stlat-lat_ami(i,j))/pixel_size )+1

         IF (geo_viirs_snow(geo_index_x_ami(i,j),geo_index_y_ami(i,j)) == -99) CYCLE
  
         IF (geo_viirs_snow(geo_index_x_ami(i,j),geo_index_y_ami(i,j)) == 1) THEN
            snow_ami(i,j) = 1  !! Snow
         ELSE IF (geo_viirs_snow(geo_index_x_ami(i,j),geo_index_y_ami(i,j)) == 2) THEN
            snow_ami(i,j) = 2  !! Bad Snow
         ELSE IF (geo_viirs_snow(geo_index_x_ami(i,j),geo_index_y_ami(i,j)) == 0) THEN
            snow_ami(i,j) = 0  !! No Snow
         ELSE IF (geo_viirs_snow(geo_index_x_ami(i,j),geo_index_y_ami(i,j)) == 3) THEN
            snow_ami(i,j) = 3  !! Cloud
         ENDIF         
 
      ENDDO
      IF ( MOD(j,1000) == 0 ) PRINT*, j, ' / 5500 '
   ENDDO

   OPEN(99,FILE=TRIM(output_path)//'VIIRS.SC_AMI_Map_2km_'//date//time//'.bin', &
           ACCESS='DIRECT',RECL=5500*5500*1)
   WRITE(99,REC=1) snow_ami
   CLOSE(99)

   PRINT*, ' [Convert VIIRS Snow Data to AHI Map Complete] '

   PRINT*, ' VIIRS File : ', date, time, ' / Complete!! '
   PRINT*, ' ------------------------------------------------ '

   IF ( ALLOCATED(viirs_data) ) DEALLOCATE(viirs_data)
   IF ( ALLOCATED( lat_viirs) ) DEALLOCATE( lat_viirs)
   IF ( ALLOCATED( lon_viirs) ) DEALLOCATE( lon_viirs)
   IF ( ALLOCATED(geo_index_x_viirs) ) DEALLOCATE(geo_index_x_viirs)
   IF ( ALLOCATED(geo_index_y_viirs) ) DEALLOCATE(geo_index_y_viirs)

ENDDO

!!================================================================================
ENDDO
!!================================================================================
CLOSE(11)

END PROGRAM

!!=======================================================
SUBROUTINE output_folder(output_path)

IMPLICIT NONE

CHARACTER(LEN=256), INTENT(IN)          :: output_path
LOGICAL                                 :: dir_e

INQUIRE(FILE=TRIM(output_path), EXIST=dir_e)

IF (dir_e .ne. .True.) THEN
   PRINT*, ' [ No Folder : '//TRIM(output_path)//' ]'
   CALL system('mkdir '//TRIM(output_path))
   PRINT*, ' [ Complete the Folder ] '
ENDIF

END SUBROUTINE

!!=======================================================
SUBROUTINE cal_date(year, jday, date)
IMPLICIT NONE

INTEGER(KIND=2),INTENT(IN)              :: jday, year
CHARACTER(LEN=8),INTENT(OUT)            :: date

INTEGER(KIND=2)                         :: month_day(12), tday, i, month, day
INTEGER(KIND=1)                         :: flag
CHARACTER(LEN=2)                        :: strmonth, strday
CHARACTER(LEN=4)                        :: stryear

DATA month_day /31, 28, 31, 30, 31, 30, &
                31, 31, 30, 31, 30, 31/
tday = 0  ;  i = 1

IF (MOD(year,   4) .eq. 0) flag = 1
IF (MOD(year, 100) .eq. 0) flag = 0
IF (MOD(year, 400) .eq. 0) flag = 1

IF (flag .eq. 1) month_day(2) = 29

DO WHILE(i .lt. 13)
        tday = tday + month_day(i)
        !print*, i,'th'
        !print*, tday, 'total'
        IF (jday .le. month_day(1)) THEN
                month = 1
                day   = jday
                i     = 13
        ELSE IF (jday .gt. tday .AND. jday .le. tday + month_day(i+1)) THEN
                month = i + 1
                day   = jday - (tday)
                i     = 13
        ELSE IF (jday .gt. 365) THEN
                month = 12
                day   = 31
                i     = 13
        ENDIF
        i = i + 1
ENDDO

WRITE(strday, '(i2.2)') day
WRITE(strmonth, '(i2.2)') month
WRITE(stryear, '(i4)') year

date=stryear//strmonth//strday

RETURN
END SUBROUTINE

