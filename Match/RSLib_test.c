#include "stdio.h"
#include "stdlib.h"
#include "memory.h"
#include "math.h"

int Granule2GridForBRDF(float **IN, float **LON, float **LAT, int nRow1, int nCol1, float fillvalue1, float **OUT, float **COUNT,float **COUNTFILL, float lon0, float lat0, float dxy, int nRow2, int nCol2, float fillvalue2)
{
	int i, j, ii, jj, bAdd = 0;
	float v, lon, lat;
	
	memset((float *)COUNTFILL, 0, nRow2*nCol2*sizeof(float));
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
			if(fillvalue1-0.00001<v && v<fillvalue1+0.00001){
				*((float *)COUNTFILL+ii*nCol2+jj) += 1;
				continue;
			}
			*((float *)OUT+ii*nCol2+jj) += v;
			*((float *)COUNT+ii*nCol2+jj) += 1;
		}//j
	}//i

	for(i=0; i<nRow2; i++)
	{
		for(j=0; j<nCol2; j++)
		{
			// 存在无效值，用缺省值填充
			if((*((float *)COUNTFILL+i*nCol2+j))>1)
				*((float *)OUT+i*nCol2+j) = fillvalue2;
			// 计算平均值
			else
				*((float *)OUT+i*nCol2+j) /= *((float *)COUNT+i*nCol2+j);//out�ڲ�����ʱ����ͬһ��λ�ÿ����кܶ��������count�ĸ����൱������ƽ��ֵ
			
		}//j
	}//i

	return 1;
}

int Granule2GridForFill(float **IN, float **LON, float **LAT, int nRow1, int nCol1, float fillvalue1, float **OUT, float **COUNT,float **COUNTFILL, float lon0, float lat0, float dxy, int nRow2, int nCol2, float fillvalue2)
{
	int i, j, ii, jj, bAdd = 0;
	float v, lon, lat;
	
	memset((float *)COUNTFILL, 0, nRow2*nCol2*sizeof(float));
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
			if(fillvalue1-0.00001<v && v<fillvalue1+0.00001){
				*((float *)COUNTFILL+ii*nCol2+jj) += 1;
				continue;
			}
			*((float *)OUT+ii*nCol2+jj) += v;
			*((float *)COUNT+ii*nCol2+jj) += 1;
		}//j
	}//i

	for(i=0; i<nRow2; i++)
	{
		for(j=0; j<nCol2; j++)
		{
			// 有效值过少，用缺省值填充
			if(((*((float *)COUNT+i*nCol2+j))/(*((float *)COUNTFILL+i*nCol2+j)))<1.3)
				*((float *)OUT+i*nCol2+j) = fillvalue2;
			// 计算平均值
			if(((*((float *)COUNT+i*nCol2+j))/(*((float *)COUNTFILL+i*nCol2+j)))>=1.3)
				*((float *)OUT+i*nCol2+j) /= *((float *)COUNT+i*nCol2+j);//out�ڲ�����ʱ����ͬһ��λ�ÿ����кܶ��������count�ĸ����൱������ƽ��ֵ
			
		}//j
	}//i

	return 1;
}

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