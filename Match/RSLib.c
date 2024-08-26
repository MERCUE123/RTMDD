#include "stdio.h"
#include "stdlib.h"
#include "memory.h"
#include "mycomplex.h"

float azimuth(float lon1, float lat1, float lon2, float lat2)
{//input longitude & latitude in degree(s)
    float result, c, v;
	int ilat1, ilat2, ilon1, ilon2;
	
    ilat1 = (int) (0.50 + lat1 * 360000.0);
    ilat2 = (int) (0.50 + lat2 * 360000.0);
    ilon1 = (int) (0.50 + lon1 * 360000.0);
    ilon2 = (int) (0.50 + lon2 * 360000.0);
    lat1 = PI * lat1 / 180;		/*degree to radian*/
    lon1 = PI * lon1 / 180;
    lat2 = PI * lat2 / 180;
    lon2 = PI * lon2 / 180;

    if ((ilat1 == ilat2) && (ilon1 == ilon2))      return 0.0F;
	else if (ilon1 == ilon2)
	{
        if (ilat1 > ilat2)       result = 180.0F;
		else return 0.0F;
    }
	else
	{	
		v = sin(lat2) * sin(lat1) + cos(lat2) * cos(lat1) * cos((lon2 - lon1));
		if(v > 1.0)	v = 1.0f;
		if(v < -1.0) v = -1.0f;
        c = acos(v);
		v = cos(lat2) * sin((lon2 - lon1)) / sin(c);
		if(v > 1.0)	v = 1.0f;
		if(v < -1.0) v = -1.0f;
        result = (float)(180 * asin(v) / PI);		//in degree
        if ((ilat2 >= ilat1) && (ilon2 > ilon1))
		{
        }
		else if (ilat2 < ilat1)
			result = 180.0F - result;
		/*else if ((ilat2 > ilat1) && (ilon2 < ilon1))
		{
			if(result < 0)	result += 360.0F;
		}*/
		
		if(result < 0)	result += 360.0F;
    }

    return result;
}

int computeAzimuth(float **LON, float **LAT, float **SC_LON, float **SC_LAT, int nRow, int nCol, float fillvalue1, float **OUT, float fillvalue2)
{
	int i, j;
	
	for(i=0; i< nRow; i++)
	{
		for(j=0; j<nCol; j++)
		{
			if((fillvalue1-0.00001<*((float *)LON+i*nCol+j) && *((float *)LON+i*nCol+j)<fillvalue1+0.00001) || (fillvalue1-0.00001<*((float *)SC_LON+i*nCol+j) && *((float *)SC_LON+i*nCol+j)<fillvalue1+0.00001))
				*((float *)OUT+i*nCol+j) = fillvalue2;
			else
				*((float *)OUT+i*nCol+j) = azimuth(*((float *)LON+i*nCol+j), *((float *)LAT+i*nCol+j), *((float *)SC_LON + i*nCol+j), *((float *)SC_LAT + i*nCol+j));
		}//j
	}//i
	return 1;
}

//Resample granule data into regular grid space
int Granule2Grid(float **IN, float **LON, float **LAT, int nRow1, int nCol1, float fillvalue1, 
float **OUT, float **COUNT, float lon0, float lat0, float dxy, int nRow2, int nCol2, float fillvalue2)
{
	int i, j, ii, jj, bAdd = 0;
	float v, lon, lat;
	memset((float *)COUNT, 0, nRow2*nCol2*sizeof(float));
	for(i=0; i<nRow2; i++)
	{
		for(j=0; j<nCol2; j++)
			*((float *)OUT+i*nCol2+j) = fillvalue2;
	}//i
	
	if(lon0>=0 && lon0+(nCol2-0.5)*dxy > 180) bAdd = 1;		//longitude from 0 to 360
	for(i=0; i<nRow1; i++)
	{
		for(j=0; j<nCol1; j++)
		{
			v = *((float *)IN + i*nCol1+j);
			if(fillvalue1-0.00001<v && v<fillvalue1+0.00001)	continue;
			//if(v == fillvalue1)	continue;

			lon = *((float *)LON+i*nCol1+j);
			if(bAdd==1 && lon<0)	lon += 360;		//longitude from 0 to 360
			lat = *((float *)LAT+i*nCol1+j);

			ii = (int)((lat0 - lat) / dxy + 0.5);
			if(ii<0 || ii>=nRow2)	continue;
			jj = (int)((lon - lon0) / dxy + 0.5);
			if(jj<0 || jj>=nCol2)	continue;
			
			/*if(ii==30 && jj==8)
			{
				FILE *fp = fopen("D:\\1.txt", "a+");
				fprintf(fp, "(%d,%d): %.2f\n", i,j,v);
				fclose(fp);
			}*/

			*((float *)OUT+ii*nCol2+jj) += v;
			*((float *)COUNT+ii*nCol2+jj) += 1;
		}//j
	}//i

	for(i=0; i<nRow2; i++)
	{
		for(j=0; j<nCol2; j++)
		{
			if(*((float *)COUNT+i*nCol2+j)>0)
				*((float *)OUT+i*nCol2+j) /= *((float *)COUNT+i*nCol2+j);//out�ڲ�����ʱ����ͬһ��λ�ÿ����кܶ��������count�ĸ����൱������ƽ��ֵ
		}//j
	}//i

	return 1;
}

int Granule2GridforVAA(float **IN, float **LON, float **LAT, int nRow1, int nCol1, float fillvalue1, float **OUT, float lon0, float lat0, float dxy, int nRow2, int nCol2, float fillvalue2)
{
	int i, j, ii, jj, bAdd = 0;
	float v, lon1, lat1, lon2, lat2, dist, *DIST;

	DIST = (float *) malloc(nRow2*nCol2*sizeof(float));
	if(DIST==NULL)	return 0;
	for(i=0; i<nRow2; i++)
	{
		for(j=0; j<nCol2; j++)
		{
			DIST[i*nCol2+j] = 999;
			*((float *)OUT+i*nCol2+j) = fillvalue2;
		}
	}//i
	
	if(lon0>=0 && lon0+(nCol2-0.5)*dxy > 180) bAdd = 1;		//longitude from 0 to 360
	for(i=0; i<nRow1; i++)
	{
		for(j=0; j<nCol1; j++)
		{
			v = *((float *)IN + i*nCol1+j);
			if(fillvalue1-0.00001<v && v<fillvalue1+0.00001)	continue;

			lon1 = *((float *)LON+i*nCol1+j);
			if(bAdd==1 && lon1<0)	lon1 += 360;		//longitude from 0 to 360
			lat1 = *((float *)LAT+i*nCol1+j);

			ii = (int)((lat0 - lat1) / dxy + 0.5);
			if(ii<0 || ii>=nRow2)	continue;
			jj = (int)((lon1 - lon0) / dxy + 0.5);
			if(jj<0 || jj>=nCol2)	continue;
			
			lon2 = lon0 + ii*dxy;
			lat2 = lat0 - ii*dxy;
			dist = fabs(lon2-lon1) + fabs(lat2-lat1);
			if(dist < DIST[ii*nCol2+jj])
			{
				*((float *)OUT+ii*nCol2+jj) = v;
				DIST[ii*nCol2+jj] = dist;
			}
		}//j
	}//i
	
	free(DIST);
	return 1;
}


//Resample regular grid data into granule space by linear or bi-linear interpolation
//Granule2Grid(float **IN, float **LON, float **LAT, int nRow1, int nCol1, float fillvalue1, float **OUT, float **COUNT, float lon0, float lat0, float dxy, int nRow2, int nCol2, float fillvalue2)
int Grid2Granule(float **IN, float lon0, float lat0, float dxy, int nRow1, int nCol1, float fillvalue, float **OUT, float **LON, float **LAT, int nRow2, int nCol2)
{
	int i, j, ii, jj, bAdd=0;
	float v1, v2, x, y, lon, lat;

	if(lon0>=0 && lon0+(nCol1-0.5)*dxy>180)		bAdd = 1;

	for(i=0; i<nRow2; i++)
	{
		for(j=0; j<nCol2; j++)
		{
			lat = *((float *)LAT + i*nCol2 + j);
			x = (lat0 - lat) / dxy;
			ii = (int)(x);
			if(ii<0 || ii>nRow1-1)
			{
				*((float *)OUT + i*nCol2 + j) = fillvalue;
				continue;
			}

			lon = *((float *)LON + i*nCol2 + j);
			if(bAdd==1 && lon<0)	lon += 360;		//longitude from 0 to 360
			y = (lon - lon0) / dxy;
			jj = (int)(y);
			if(jj<0 || jj>nCol1-1)
			{
				*((float *)OUT + i*nCol2 + j) = fillvalue;
				continue;
			}

			if(ii==nRow1-1 || jj==nCol1-1)
			{
				if(ii==nRow1-1)
				{
					if(jj==nCol1-1)	*((float *)OUT + i*nCol2 + j) = *((float *)IN + ii*nCol1+jj);
					else
					{
						if(*((float *)IN + ii*nCol1+jj+1)==fillvalue || *((float *)IN + ii*nCol1+jj)==fillvalue)
							*((float *)OUT + i*nCol2 + j) = fillvalue;
						else		*((float *)OUT + i*nCol2 + j) = (y-jj)*(*((float *)IN + ii*nCol1+jj+1)) + (jj+1-y)*(*((float *)IN + ii*nCol1+jj)); //linear interpolation
					}
				}
				else /*jj==nCol1-1*/
				{
					if(*((float *)IN + (ii+1)*nCol1+jj)==fillvalue || *((float *)IN + ii*nCol1+jj)==fillvalue)
						*((float *)OUT + i*nCol2 + j) = fillvalue;
					else		*((float *)OUT + i*nCol2 + j) = (x-ii)*(*((float *)IN + (ii+1)*nCol1+jj)) + (ii+1-x)*(*((float *)IN + ii*nCol1+jj)); //linear interpolation
				}

			}
			else	/*bi-linear interpolation*/
			{
				if(*((float *)IN + (ii+1)*nCol1+jj)==fillvalue || *((float *)IN +ii*nCol1+jj)==fillvalue || *((float *)IN + (ii+1)*nCol1+jj+1)==fillvalue || *((float *)IN + ii*nCol1+jj+1)==fillvalue)
					*((float *)OUT + i*nCol2 + j) = fillvalue;
				else
				{
					v1 = (x-ii)*(*((float *)IN + (ii+1)*nCol1+jj)) + (ii+1-x)*(*((float *)IN +ii*nCol1+jj));
					v2 = (x-ii)*(*((float *)IN + (ii+1)*nCol1+jj+1)) + (ii+1-x)*(*((float *)IN + ii*nCol1+jj+1));
					*((float *)OUT + i*nCol2 + j) = (y-jj)*v2 + (jj+1-y)*v1;
				}
			}//else
		}//j
	}//i

	return 1;
}

//Resample ERA5 data (regular grid data) into granule space
int ERA5Var2Granule(float **LON, float **LAT, int nRow, int nCol, int nStart, int nEnd, float **VAR, float **OUT)
{
	int i, j, ii, jj, nRow_ERA5=721, nCol_ERA5=1440;
	float lon0=0, lat0=90, dxy=0.25f, v1, v2, x, y, lon, lat;

	for(i=nStart; i<nEnd; i++)
	{
		for(j=0; j<nCol; j++)
		{
			lat = *((float *)LAT + i*nCol + j);
			x = (lat0 - lat) / dxy;
			ii = (int)(x);
			if(ii<0 || ii>nRow_ERA5-1)		return 0;

			lon = *((float *)LON + i*nCol + j);
			if(lon < 0)	lon += 360;
			y = (lon - lon0) / dxy;
			jj = (int)(y);
			if(jj<0 || jj>nCol_ERA5-1)	return 0;

			if(ii==nRow_ERA5-1 || jj==nCol_ERA5-1)
			{
				if(ii==nRow_ERA5-1)
				{
					if(jj==nCol_ERA5-1)	*((float *)OUT + i*nCol + j) = *((float *)VAR + ii*nCol_ERA5+jj);
					else		*((float *)OUT + i*nCol + j) = (y-jj)*(*((float *)VAR + ii*nCol_ERA5+jj+1)) + (jj+1-y)*(*((float *)VAR + ii*nCol_ERA5+jj));
				}
				else /*jj==nCol_ERA5-1*/
					*((float *)OUT + i*nCol + j) = (x-ii)*(*((float *)VAR + (ii+1)*nCol_ERA5+jj)) + (ii+1-x)*(*((float *)VAR + ii*nCol_ERA5+jj));

			}
			else
			{
				v1 = (x-ii)*(*((float *)VAR + (ii+1)*nCol_ERA5+jj)) + (ii+1-x)*(*((float *)VAR +ii*nCol_ERA5+jj));
				v2 = (x-ii)*(*((float *)VAR + (ii+1)*nCol_ERA5+jj+1)) + (ii+1-x)*(*((float *)VAR + ii*nCol_ERA5+jj+1));
				*((float *)OUT + i*nCol + j) = (y-jj)*v2 + (jj+1-y)*v1;
			}
		}//j
	}//i

	return 1;
}


//Total absorption by oxygen and water vapor -- clear-sky conditions
double MPM93(double freq, double pressure, double temperature, double RH)
{	//freq-Frequency in GHz
	//pressure in hPa (mba)
	//temperature in Kelvin
	//RH is the relative humidity (%)
	//Return: Complex Refractivity in ppm
	struct complex ZN, ZNw, ZNr, ZEp, ZH[38], ZNN, ZNN1;
	double TH, THV, E, Y, Es, P, GSP, AP1, AP2, Q, Q1, Q2, Q3, FQ, AGD, EN0;
	double AH[79];
	int J;

	ZN.rmz=ZN.imz=0.0;
	if(freq<=0 || freq>1000 || pressure<=0 || pressure>1100 || temperature<173.15 || temperature>323.15 || RH<0)	//if exceed the valid range, failed
		return -1;

	double BFFS[79] = {50.474238, 50.987749, 51.503350, 52.021410, 52.542394,
					   53.066907, 53.595749, 54.130000, 54.671159, 55.221367,
					   55.783802, 56.264775, 56.363389, 56.968206, 57.612484,
					   58.323877, 58.446590, 59.164207, 59.590983, 60.306061,
					   60.434776, 61.150560, 61.800154, 62.411215, 62.486260,
					   62.997977, 63.568518, 64.127767, 64.678903, 65.224071,
					   65.764772, 66.302091, 66.836830, 67.369598, 67.900867,
					   68.431005, 68.960311,118.750343,368.498350,424.763124,
					   487.249370,715.393150,773.839675,834.145330,
					   22.235080, 67.803960,119.995940,183.310091,321.225644,
					   325.152919,336.222601,380.197372,390.134508,437.346667,
					   439.150812,443.018295,448.001075,470.888947,474.689127,
					   488.491133,503.568532,504.482692,547.676440,552.020960,
					   556.936002,620.700807,645.866155,658.005280,752.033227,
					   841.053973,859.962313,899.306675,902.616173,906.207325,
					   916.171582,923.118427,970.315022,987.926764,1780.00000};
	double A1[79] = {0.094E-6,   0.246E-6,   0.608E-6,   1.414E-6,   3.102E-6,
					 6.410E-6,  12.470E-6,  22.800E-6,  39.180E-6,  63.160E-6,
					 95.350E-6,  54.890E-6, 134.400E-6, 176.300E-6, 214.100E-6,
					 238.600E-6, 145.700E-6, 240.400E-6, 211.200E-6, 212.400E-6,
					 246.100E-6, 250.400E-6, 229.800E-6, 193.300E-6, 151.700E-6,
					 150.300E-6, 108.700E-6,  73.350E-6,  46.350E-6,  27.480E-6,
					 15.300E-6,   8.009E-6,   3.946E-6,   1.832E-6,   0.801E-6,
					 0.330E-6,   0.128E-6,  94.500E-6,   6.790E-6,  63.800E-6,
					 23.500E-6,   9.960E-6,  67.100E-6,  18.000E-6,
					 0.1130E-1,  0.0012E-1,  0.0008E-1,  2.4200E-1,  0.0483E-1,
					 1.4990E-1,  0.0011E-1, 11.5200E-1,  0.0046E-1,  0.0650E-1,
					 0.9218E-1,  0.1976E-1, 10.3200E-1,  0.3297E-1,  1.2620E-1,
					 0.2520E-1,  0.0390E-1,  0.0130E-1,  9.7010E-1, 14.7700E-1, 
					 487.4000E-1,  5.0120E-1,  0.0713E-1,  0.3022E-1,239.6000E-1,  
					 0.0140E-1,  0.1472E-1,  0.0605E-1,  0.0426E-1,  0.1876E-1,  
					 8.3410E-1,  0.0869E-1,  8.9720E-1,132.1000E-1,22300.0000E-1};
	double A2[79] = {9.694, 8.694, 7.744, 6.844, 6.004,
					 5.224, 4.484, 3.814, 3.194, 2.624,
					 2.119, 0.015, 1.660, 1.260, 0.915,
					 0.626, 0.084, 0.391, 0.212, 0.212,
					 0.391, 0.626, 0.915, 1.260, 0.083,
					 1.665, 2.115, 2.620, 3.195, 3.815,
					 4.485, 5.225, 6.005, 6.845, 7.745,
					 8.695, 9.695, 0.009, 0.049, 0.044,
					 0.049, 0.145, 0.130, 0.147,
					 2.143, 8.735, 8.356, 0.668, 6.181,
					 1.540, 9.829, 1.048, 7.350, 5.050,
					 3.596, 5.050, 1.405, 3.599, 2.381,
					 2.853, 6.733, 6.733, 0.114, 0.114,
					 0.159, 2.200, 8.580, 7.820, 0.396, 
					 8.180, 7.989, 7.917, 8.432, 5.111, 
					 1.442,10.220, 1.920, 0.258, 0.952};
	double A3[79] = {0.890E-3,  0.910E-3,  0.940E-3,  0.970E-3,  0.990E-3,
					 1.020E-3,  1.050E-3,  1.070E-3,  1.100E-3,  1.130E-3,
					 1.170E-3,  1.730E-3,  1.200E-3,  1.240E-3,  1.280E-3,
					 1.330E-3,  1.520E-3,  1.390E-3,  1.430E-3,  1.450E-3,
					 1.360E-3,  1.310E-3,  1.270E-3,  1.230E-3,  1.540E-3,
					 1.200E-3,  1.170E-3,  1.130E-3,  1.100E-3,  1.070E-3,
					 1.050E-3,  1.020E-3,  0.990E-3,  0.970E-3,  0.940E-3,
					 0.920E-3,  0.900E-3,  1.630E-3,  1.920E-3,  1.930E-3,
					 1.920E-3,  1.810E-3,  1.810E-3,  1.820E-3,
					 2.811E-3,  2.858E-3,  2.948E-3,  3.050E-3,  2.303E-3,
					 2.783E-3,  2.693E-3,  2.873E-3,  2.152E-3,  1.845E-3,
					 2.100E-3,  1.860E-3,  2.632E-3,  2.152E-3,  2.355E-3,
					 2.602E-3,  1.612E-3,  1.612E-3,  2.600E-3,  2.600E-3,
					 3.210E-3,  2.438E-3,  1.800E-3,  3.210E-3,  3.060E-3, 
					 1.590E-3,  3.060E-3,  2.985E-3,  2.865E-3,  2.408E-3, 
					 2.670E-3,  2.900E-3,  2.550E-3,  2.985E-3, 17.620E-3};
	double A4[79] = {0.0,  0.0,  0.0,  0.0,  0.0,
					 0.0,  0.0,  0.0,  0.0,  0.0,
					 0.0,  0.0,  0.0,  0.0,  0.0,
					 0.0,  0.0,  0.0,  0.0,  0.0,
					 0.0,  0.0,  0.0,  0.0,  0.0,
					 0.0,  0.0,  0.0,  0.0,  0.0,
					 0.0,  0.0,  0.0,  0.0,  0.0,
					 0.0,  0.0,  0.0,  0.0,  0.0,
					 0.0,  0.0,  0.0,  0.0,
					 4.80,  4.93,  4.78,  5.30,  4.69,
					 4.85,  4.74,  5.38,  4.81,  4.23,
					 4.29,  4.23,  4.84,  4.57,  4.65,
					 5.04,  3.98,  4.01,  4.50,  4.50,
					 4.11,  4.68,  4.00,  4.14,  4.09,  
					 5.76,  4.09,  4.53,  5.10,  4.70,  
					 5.00,  4.78,  4.94,  4.55, 30.50};
	double A5[79] = {0.240E-3,  0.220E-3,  0.197E-3,  0.166E-3,  0.136E-3,
					 0.131E-3,  0.230E-3,  0.335E-3,  0.374E-3,  0.258E-3,
					-0.166E-3,  0.390E-3, -0.297E-3, -0.416E-3, -0.613E-3,
					-0.205E-3,  0.748E-3, -0.722E-3,  0.765E-3, -0.705E-3,
					 0.697E-3,  0.104E-3,  0.570E-3,  0.360E-3, -0.498E-3,
					 0.239E-3,  0.108E-3, -0.311E-3, -0.421E-3, -0.375E-3,
					-0.267E-3, -0.168E-3, -0.169E-3, -0.200E-3, -0.228E-3,
					-0.240E-3, -0.250E-3, -0.036E-3,  0, 0, 0, 0, 0, 0,
					 0.69,  0.69,  0.70,  0.64,  0.67,
					 0.68,  0.69,  0.54,  0.63,  0.60,
					 0.63,  0.60,  0.66,  0.66,  0.65,
					 0.69,  0.61,  0.61,  0.70,  0.70,
					 0.69,  0.71,  0.60,  0.69,  0.68,  
					 0.33,  0.68,  0.68,  0.70,  0.70,  
					 0.70,  0.70,  0.64,  0.68,  2.00};
	double A6[79] = {0.790E-3,  0.780E-3,  0.774E-3,  0.764E-3,  0.751E-3,
					 0.714E-3,  0.584E-3,  0.431E-3,  0.305E-3,  0.339E-3,
					 0.705E-3, -0.113E-3,  0.753E-3,  0.742E-3,  0.697E-3,
					 0.051E-3, -0.146E-3,  0.266E-3, -0.090E-3,  0.081E-3,
					-0.324E-3, -0.067E-3, -0.761E-3, -0.777E-3,  0.097E-3,
					-0.768E-3, -0.706E-3, -0.332E-3, -0.298E-3, -0.423E-3,
					-0.575E-3, -0.700E-3, -0.735E-3, -0.744E-3, -0.753E-3,
					-0.760E-3, -0.765E-3,  0.009E-3, 0, 0, 0, 0, 0, 0,
					 1.00,  0.82,  0.79,  0.85,  0.54,
					 0.74,  0.61,  0.89,  0.55,  0.48,
					 0.52,  0.50,  0.67,  0.65,  0.64,
					 0.72,  0.43,  0.45,  1.00,  1.00,
					 1.00,  0.68,  0.50,  1.00,  0.84,  
					 0.45,  0.84,  0.90,  0.95,  0.53,  
					 0.78,  0.80,  0.67,  0.90,  5.00};
	temperature -= 273.15;	//Kelvin to degree celsius
	ZNN.rmz = ZNN.imz = 0.0;
	TH=300.0/(temperature+273.15);
	THV=log(TH);
	E=RH;
	if(E<0)	E = 0;
	if(temperature>-40)
	{
		Y = 373.16/(temperature+273.16);
		Es = -7.90298*(Y-1.0) + 5.02808*log10(Y) - 1.3816E-7*(pow(10,11.344*(1.0-(1.0/Y))) -1.0) + 8.1328E-3*(pow(10,-3.49149*(Y-1.0))-1.0)+log10(1013.246);
	}
	else
	{
		Y=273.16/(temperature+273.16);
		Es=-9.09718*(Y-1.0)-3.56654*log10(Y)+ 0.876793*(1.0-(1.0/Y))+log10(6.1071);
	}
	Es=pow(10,Es);
	E=Es*E/100;
	P=pressure-E;
    EN0=(0.2588*P+(0.239+4.163*TH)*E)*TH;	//constant component X10? different from the expression in MPM89

	//Oxygen
	GSP=(P+E)*exp(THV*0.8)*0.56E-3;
    AP1=6.14E-5*P*pow(TH,2);
    AP2=pow(P,2)*exp(3.5*THV-27.2945);
    Q1=P*exp(3.0*THV);
    Q3=pressure*exp(0.8*THV);
    Q2=P*exp(0.8*THV)+1.1*E*TH;

	FQ=freq;
	for(J=0;J<38;J++)
	{
		Q=Q1*A1[J]*exp(A2[J]*(1.0-TH));
        ZH[J]=multiplyComplex(constructComplex(Q/BFFS[J], 0), constructComplex(1.0, -(A5[J]+A6[J]*TH)*Q3));							//CMPLX
        Q=sqrt(pow(A3[J]*Q2,2)+2.25E-6);
		ZNN1 = minusComplex(divideComplex(ZH[J], constructComplex(BFFS[J]-freq, -Q)), conjugateComplex(divideComplex(ZH[J], constructComplex(BFFS[J]+freq,-Q))));
        ZNN=addComplex(ZNN, ZNN1);					//CMPLX
	}
	Q2=P*exp(0.2*THV)+1.1*E*TH;
	for(J=38;J<44;J++)
	{
		Q=Q1*A1[J]*exp(A2[J]*(1.0-TH));
        AH[J]=Q/BFFS[J];
        Q=sqrt(pow(A3[J]*Q2,2)+2.25E-6);
		ZNN1 = minusComplex(divideComplex(constructComplex(AH[J], 0), constructComplex(BFFS[J]-freq,-Q)), divideComplex(constructComplex(AH[J], 0), constructComplex(BFFS[J]+freq, Q)));
        ZNN=addComplex(ZNN, ZNN1);															//CMPLX
	}//End of 

	if(ZNN.imz<0)	ZNN.rmz = ZNN.imz = 0.0;
	ZNN1 = minusComplex(ZNN, divideComplex(constructComplex(AP1, 0), constructComplex(FQ, GSP)));
	ZNN=addComplex(ZNN1, constructComplex(0.0,AP2/(1.0+1.93E-5*pow(FQ,1.5))));				//CMPLX

	//Water vapor
	AGD=2.1316E-12/TH;
	Q1=E*exp(3.5*THV);
	for(J=44;J<79;J++)
	{
		Q=Q1*A1[J]*exp(A2[J]*(1.0-TH));
        AH[J]=Q/BFFS[J];
        Q=A3[J]*(P*exp(A5[J]*THV)+E*A4[J]*exp(A6[J]*THV));
        Q=0.535*Q+sqrt(0.217*Q*Q+AGD*pow(BFFS[J],2));
		ZNN1 = minusComplex(divideComplex(constructComplex(AH[J], 0), constructComplex(BFFS[J]-freq, -Q)), divideComplex(constructComplex(AH[J], 0), constructComplex(BFFS[J]+freq, Q)));
        ZNN=addComplex(ZNN,ZNN1);				//CMPLX
	}//End of J

	ZN=multiplyComplex(constructComplex(FQ, 0), ZNN);
	//return	addComplex(constructComplex(EN0, 0), ZN);			//complex refractivity in ppm
	//return 0.1820*freq*ZN.imz;		//Absorption coefficient in dB/km
	return 0.04191*freq*ZN.imz;		//Absorption coefficient in Nepers/km
}

double MPM93_Clouds(double freq, double rho, int bIce, double RainRate)
{	//Compute the absorption ONLY due to clouds and rain!!!
	//freq-Frequency in GHz
	//pressure in hPa (mba)
	//temperature in Kelvin
	//RH is the relative humidity (%)
	//rho-Water mass density in g/m^3 when the relative humidity exceeds saturation, RH=100-101
	//bIce=1, ice permittivity model; bIce=0, water permittivity model
	//RainRate, rain rate in mm/hr
	//Return: Complex Refractivity in ppm
	struct complex ZN, ZNw, ZNr, ZEp;
	double TH;

	//Complex refractivity due to suspended particles
	ZNw.rmz = ZNw.imz = 0.0;	//Cloud/Fog module
	if(rho>0)					//water mass density is greater than 0
	{
		double fD,fS,Eps,Epinf,Eopt,Ai,Bi;
		if(bIce==0)	//Water permittivity
		{
			fD = 20.20-146.4*(TH-1)+316*pow(TH-1,2);
			fS = 39.8*fD;
			Eps=103.3*(TH-1)+77.66;
			Epinf=0.0671*Eps;
			Eopt=3.52;
			//Complex Permittivity of water (double-Debye model)
			ZEp = addComplex(divideComplex(constructComplex(Eps-Epinf, 0), constructComplex(freq, fD)), divideComplex(constructComplex(Epinf-Eopt, 0), constructComplex(freq, fS)));
			ZEp.rmz *= freq;
			ZEp.imz *= freq;
			ZEp = minusComplex(constructComplex(Eps, 0), ZEp);
		}
		else	//Ice permittivity [8]
		{
			Ai=(62.0*TH-11.6)*1.0E-4*exp(-22.1*(TH-1.0));
			Bi=0.542E-6*(-24.17+116.79/TH+pow(TH/(TH-0.9927),2.0));
			Eps=3.15;
			//Complex Permittivity of Ice
			if(freq<0.001)	freq=0.001;
			ZEp=constructComplex(3.15, Ai/freq+Bi*freq);
		}

		//Suspended particle Rayleigh approximation [6]
		ZNw = divideComplex(minusComplex(ZEp, constructComplex(1.0, 0)), addComplex(ZEp, constructComplex(2.0, 0)));
		ZNw.rmz = 1.5*rho*( ZNw.rmz - 1.0 + 3.0/(Eps+2.0) );
		ZNw.imz = 1.5*rho*ZNw.imz;
	}

	//Complex refractivity due to rain
	ZNr.rmz=ZNr.imz=0.0;
	if(RainRate>0)
	{
		double ARAIN=0.0,BRAIN=0.0,ATRAN=0.0,Nrp=0.0,GA,EA,GB,EB;
		double fr,Nro,X;

		//Alpha calculation
		if(freq<2.9)
		{
			GA=6.39E-5;
			EA=2.03;
		}
		else if(freq<54)
		{
			GA=4.21E-5;
			EA=2.42;
		}
		else if(freq<180)
		{
			GA=4.09E-2;
			EA=0.699;
		}
		else
		{
			GA=3.38;
			EA=-0.151;
		}
		ARAIN=GA*pow(freq,EA);

		//Beta calculation
		if(freq<8.5)
		{
			GB=0.851;
			EB=0.158;
		}
		else if(freq<25)
		{
			GB=1.41;
			EB= -0.0779;
		}
		else if(freq<164)
		{
			GB=2.63;
			EB= -0.272;
		}
		else
		{
			GB=0.616;
			EB=0.0126;
		}
		BRAIN=GB*pow(freq,EB);
		ATRAN=ARAIN*pow(RainRate,BRAIN);

		//Rain delay approximated after ZUFFEREY [10], who
		//used Marshall-Palmer drop size distribution and 20 deg. C
		fr=53.0-0.37*RainRate+1.5E-3*pow(RainRate,2);
		Nro=(RainRate*(3.68-0.012*RainRate))/fr;
		X=freq/fr;
		Nrp=-Nro*pow(X,2.5)/(1+pow(X,2.5));

		ZNr.rmz = Nrp;
		ZNr.imz = ATRAN/(0.1820*freq);
	}//End of RainRate>0

	//return	addComplex(addComplex(ZN, ZNw), ZNr);			//complex refractivity in ppm
	ZN = addComplex(ZNw, ZNr);
	//return 0.1820*freq*ZN.imz;		//Absorption coefficient in dB/km
	return 0.04191*freq*ZN.imz;		//Absorption coefficient in Nepers/km
}