#include <stdio.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <time.h>
#include <stdlib.h>

#define OUTPUT stdout

int gfactorNum;

//int trfread = 0;
//int tefread = 0;

//inline double Validate(FILE *testDataFile,double av,double bu , double bi , double *pu ,double *qi,int factorNum);

//can be repaced
inline int readTrainDataFile(FILE *fp,int *uid,int *iid,int *score)
{
	char *data = NULL;
	int llen;
	size_t nbytes;
	const char *delim = "\t";
	if(fp == NULL) return 0;
	if((llen = getline(&data,&nbytes,fp)) == -1) return 0;
	//printf("[%s]",data);
	*uid = atoi(strtok(data,delim)) -1; // arg[0]
	*iid = atoi(strtok(NULL,delim)) -1;//arg[1]
	*score = atoi(strtok(NULL,delim));//arg[2]
	//printf("read train data: %d %d %d\n",*uid,*iid,*score);
	//if(!trfread) getchar();
	//trfread++;
	free(data);
	return 1;
}

//can be repaced
inline int readTestDataFile(FILE *fp,int *uid,int *iid,int *score)
{
	char *data = NULL;
	int llen;
	size_t nbytes;
	const char *delim = "\t";
	if(fp == NULL) return 0;
	if((llen = getline(&data,&nbytes,fp)) == -1) return 0;
	*uid = atoi(strtok(data,delim)) -1; // arg[0]
	*iid = atoi(strtok(NULL,delim)) -1;//arg[1]
	*score = atoi(strtok(NULL,delim));//arg[2]
	//printf("read test data: %d %d %d\n",*uid,*iid,*score);
	//if(!tefread) getchar();
	//tefread++;
	free(data);
	return 1;
}

//
inline int readConfigureFile(FILE *fp,double *averageScore,int *userNum ,int *itemNum ,
			int *factorNum,double *learnRate,double *regularization)
{
	char *data = NULL;
	int llen;
	size_t nbytes;
	const char *delim = " ";
	if(fp == NULL) return 0;
	rewind(fp);
	if((llen = getline(&data,&nbytes,fp)) == -1) return 0;
	*averageScore = atof(strtok(data,delim)); // arg[0]
	*userNum = atoi(strtok(NULL,delim));//arg[1]
	*itemNum = atoi(strtok(NULL,delim));//arg[2]
	*factorNum = atoi(strtok(NULL,delim));//arg[3]
	*learnRate = atof(strtok(NULL,delim)); // arg[4]
	*regularization = atof(strtok(NULL,delim)); // arg[5]
	free(data);
	return 1;
}

// with parse
double Average(const char *fileName)
{
	FILE *fp = fopen(fileName,"r");
	char *data = NULL;
	char *tmp;
	int llen;
	size_t nbytes;
	const char *delim = " ";
	if(fp == NULL) return 0;
	double result = 0.0;
	int cnt = 0;
 	while((llen = getline(&data,&nbytes,fp))!= -1)
	{
		cnt++;
		strtok(data,delim); // arg[0]
		strtok(NULL,delim);//arg[1]
		tmp = strtok(NULL,delim);//arg[2]
		result += atof(tmp);
		free(data);
		data = NULL;
	}
	printf("%f-%f\n",result,cnt);
	fclose(fp);
	if(!cnt) return 0;
	return result / cnt;
}

inline double InerProduct(double *v1,double *v2,int lv1)
{
	int i;
	double result = 0;
	for(i = 0 ; i <lv1 ; i ++) result += v1[i] * v2[i];
	return result;
}

inline double PredictScore(double av,double bu,double bi,double *pu,double *qi,int len)
{
	double pScore = av + bu + bi + InerProduct(pu,qi,len);
	return (pScore < 1) ? ( 1 ):( (pScore > 5 )? (5) : (pScore) ); 
}

//validate the model
inline double Validate(FILE *testDataFile,double av,double *bu , double *bi , 
		double (*pu)[gfactorNum] ,double (*qi)[gfactorNum],int factorNum)
{
	double rmse = 0;
	int cnt = 0;
	int uid;
	int iid;
	int score;
	int pScore;
	int tScore;
	fseek(testDataFile,0L,SEEK_SET); //rewind(testDataFile);
	while(readTestDataFile(testDataFile,&uid,&iid,&score))
	{
		cnt ++;
		pScore = PredictScore(av,bu[uid],bi[iid],pu[uid],qi[iid],factorNum);
		tScore = score;
		rmse += (tScore - pScore) * (tScore - pScore);
	}
	//printf("~ rmse %f-- cnt %d\n",rmse,cnt);
	return sqrt(rmse / cnt);	
}



int SVD(FILE *configureFile,FILE * testDataFile,FILE *trainDataFile,FILE *modelSaveFile)
{
	//char *data = NULL;
	//size_t nbytes;
	//int llen;
	//const char *delim = " ";
	//if((llen = getline(&data,&nbytes,configureFile)) == -1) return 0;
	//double averageScore = atof(strtok(data,delim)); // arg[0]
	//int userNum = atoi(strtok(NULL,delim));//arg[1]
	//int itemNum = atoi(strtok(NULL,delim));//arg[2]
	//int factorNum = atoi(strtok(NULL,delim));//arg[3]
	//double learnRate = atof(strtok(NULL,delim)); // arg[4]
	//double regularization = atof(strtok(NULL,delim)); // arg[5]
	
	double averageScore ;
	int userNum ;
	int itemNum ;
	int factorNum ;
	double learnRate ;
	double regularization ;	
	readConfigureFile(configureFile,&averageScore,&userNum ,&itemNum ,
			&factorNum,&learnRate,&regularization);
			
	fprintf(OUTPUT,"%f-%d-%d-%d-%f-%f\n",averageScore,userNum,itemNum,
			factorNum,learnRate,regularization);
	//getchar();
	gfactorNum = factorNum;
	int i;
	int j;
	int k;
	double bi[itemNum];
	double bu[userNum];
	//memset(bi,0,itemNum*sizeof(double));
	//memset(bu,0,userNum*sizeof(double));
	for(i=0;i<itemNum;i++) bi[i] = 0;
	for(i=0;i<userNum;i++) bu[i] = 0;
	double temp = sqrt(factorNum);
	double qi[itemNum][factorNum];
	double pu[userNum][factorNum];

	double min,max;
	min = 100;
	max = 0;
	for(i=0;i<itemNum;i++)
		for(j=0;j<factorNum;j++)
		{
			qi[i][j] = 0.1*(double)(rand()%(int)(temp*10000))/10000;
			//if(i*j < 100) printf("%f --\n",qi[i][j]);
			if(qi[i][j] < min) min = qi[i][j];
			if(qi[i][j] > max) max = qi[i][j];
		}
	for(i=0;i<userNum;i++)
		for(j=0;j<factorNum;j++)
		{
			//pu[i][j] = 0.1*((double)rand()/(double)temp);
			pu[i][j] = 0.1*(double)(rand()%(int)(temp*10000))/10000;
			//if(i*j < 100) printf("%f --\n",pu[i][j]);
			
			if(pu[i][j] < min) min = pu[i][j];
			if(pu[i][j] > max) max = pu[i][j];
		}
	printf("min:%f max:%f\n",min,max);
	fprintf(OUTPUT,"initialization end\nstart training\n");
	
	//train model
	double preRmse = 1000000.0;
	int arr;
	int uid;
	int iid;
	int score;
	double eui;
	double prediction;
	double curRmse;
	
	for(i=0;i<100;i++)
	{
		rewind(trainDataFile);
		while(readTrainDataFile(trainDataFile,&uid,&iid,&score))
		{
			//printf("%d-%d-%d\n",uid,iid,score);
			prediction = PredictScore(averageScore,bu[uid],bi[iid],pu[uid],qi[iid],factorNum);
			eui = score - prediction;
			
			//update parameters
			bu[uid] += learnRate * (eui - regularization * bu[uid]);
			bi[iid] += learnRate * (eui - regularization * bi[iid]);
			
			for(k = 0;k<factorNum;k++)
			{
				//temp = pu[uid][k]	#attention here, must save the value of pu before updating
				//pu[uid][k] += learnRate * (eui * qi[iid][k] - regularization * pu[uid][k])
				//qi[iid][k] += learnRate * (eui * temp - regularization * qi[iid][k])
				
				//attention here, must save the value of pu before updating
				double temp = pu[uid][k];
				
				pu[uid][k] += learnRate * (eui * qi[iid][k] - regularization * pu[uid][k]);
				qi[iid][k] += learnRate * (eui * temp - regularization * qi[iid][k]);				
			}
		}
		
		//learnRate *= 0.9; // learnRate *= 0.9?
		curRmse = Validate(testDataFile, averageScore, bu, bi, pu, qi,factorNum);
		fprintf(OUTPUT,"preRmse :%f test_RMSE in step %d: %f\n",preRmse,i,curRmse);
		if (curRmse >= preRmse) break;
		preRmse = curRmse;
	}
	//write the model to files
	//size_t fread(void *ptr, size_t size, size_t nmemb, FILE *stream);
	//size_t fwrite(const void *ptr, size_t size, size_t nmemb,FILE *stream);

	
	for(i=0;i<itemNum;i++) fwrite(&(bi[i]),1,sizeof(double),modelSaveFile);
	for(i=0;i<userNum;i++) fwrite(&(bu[i]),1,sizeof(double),modelSaveFile);
	//for(i=0;i<itemNum;i++) fprintf(modelSaveFile,"bi[%d]=%f\n",i,bi[i]);
	//for(i=0;i<userNum;i++) fprintf(modelSaveFile,"bu[%d]=%f\n",i,bu[i]);
	for(i=0;i<itemNum;i++)
		for(j=0;j<factorNum;j++)
			fwrite(&(qi[i][j]),1,sizeof(double),modelSaveFile);
			//fprintf(modelSaveFile,"qi[%d][%d]=%f\n",i,j,qi[i][j]);
	for(i=0;i<userNum;i++)
		for(j=0;j<factorNum;j++)
			fwrite(&(pu[i][j]),1,sizeof(double),modelSaveFile);
			//fprintf(modelSaveFile,"pu[%d][%d]=%f\n",i,j,pu[i][j]);

	fprintf(OUTPUT,"model generation over\n");
	
	//free(data);
	return 1;
}

int Predict(FILE * configureFile, FILE * modelSaveFile, FILE * testDataFile, FILE * resultSaveFile)
{
	//get parameter
	double averageScore ;
	int userNum ;
	int itemNum ;
	int factorNum ;
	double learnRate ;
	double regularization ;	
	readConfigureFile(configureFile,&averageScore,&userNum ,&itemNum ,
			&factorNum,&learnRate,&regularization);
			
	fprintf(OUTPUT,"%f-%d-%d-%d-%f-%f\n",averageScore,userNum,itemNum,
			factorNum,learnRate,regularization);
			
	//get model
	double bi[itemNum];
	double bu[userNum];
	
	double qi[itemNum][factorNum];
	double pu[userNum][factorNum];
	
	int i;
	int j;
	
	for(i=0;i<itemNum;i++) fread(&(bi[i]),1,sizeof(double),modelSaveFile);
	for(i=0;i<userNum;i++) fread(&(bu[i]),1,sizeof(double),modelSaveFile);
	for(i=0;i<itemNum;i++)
		for(j=0;j<factorNum;j++)
			fread(&(qi[i][j]),1,sizeof(double),modelSaveFile);
	for(i=0;i<userNum;i++)
		for(j=0;j<factorNum;j++)
			fread(&(pu[i][j]),1,sizeof(double),modelSaveFile);
	int uid;
	int iid;
	int score;
	int pScore;
	rewind(testDataFile);
	//predict
	while(readTestDataFile(testDataFile,&uid,&iid,&score))
	{
		pScore = PredictScore(averageScore,bu[uid], bi[iid], pu[uid], qi[iid],factorNum);
		fprintf(resultSaveFile,"%f\n",pScore);
	}
	
	fprintf(OUTPUT,"predict over\n");
}
	
int main(int argc,char *argv[])
{
	FILE *configureFile = fopen("svd.conf","r");
	FILE *trainDataFile = fopen("ml_data/training.txt","r");
	FILE *testDataFile = fopen("ml_data/test.txt","r"); 
	FILE *modelSaveFile = fopen("svd_model.pkl","wb");
	//FILE *modelSaveFile = fopen("svd_model.pkl","w");
	FILE *resultSaveFile = fopen("prediction","w");
	srand((unsigned) time(NULL));
	if(configureFile == NULL || trainDataFile == NULL || testDataFile == NULL
		||modelSaveFile == NULL ||resultSaveFile == NULL)
	{
		fprintf(stderr,"error opening file %p|%p|%p|%p|%p",configureFile,trainDataFile,
					testDataFile,modelSaveFile,resultSaveFile);
		return 0;
	}
	
	// Average("svd.conf");
	// print %f Average ua.base ?
	SVD(configureFile,testDataFile,trainDataFile,modelSaveFile);
	Predict(configureFile, modelSaveFile, testDataFile, resultSaveFile);
	
	fclose(configureFile);
	fclose(trainDataFile);
	fclose(testDataFile);
	fclose(modelSaveFile);
	fclose(resultSaveFile);
	//printf("trf:%d tsf:%d \n",trfread,tefread);
	return 0;	
}


