from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np

from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN, AgglomerativeClustering
import os
np.random.seed(12)

class Distance():

    path='dataset'

    def clusteringKMeans(self,X, i):
        kmeansAtt = KMeans(n_clusters=i, max_iter=300, init='k-means++', random_state=12345).fit(X)

        return kmeansAtt.cluster_centers_

    def clusteringMiniBatchKMean(self, X, i, dim):
        kmeansAtt = MiniBatchKMeans(n_clusters=i, max_iter=300, init='k-means++', random_state=12345, init_size=dim,
                                    batch_size=50).fit(X)


        return kmeansAtt.cluster_centers_


    def clusteringAgglomerative(self, X, i):
        kmeansAtt=AgglomerativeClustering(n_clusters=i,max_iter=300, init='k-means++', random_state=12345).fit(X)

        return kmeansAtt.cluster_centers_


    def reshapeFeature(self, x):
        feature = x.reshape(1, -1)
        return feature

    def saveNpArray(self,X, Y, tipo):
        filenameX = tipo + "X.npy"
        filenameY = tipo + "Y.npy"
        np.save(os.path.join(self.path,filenameX), X)
        np.save(os.path.join(self.path,filenameY), Y)

    def loadNpArray(self):
        x = np.load(os.path.join(self.path,'FullTrainX.npy'))
        y = np.load(os.path.join(self.path,'FullTrainY.npy'))

        return x, y


    def distance(self,X_Train, X_Test, dfNormal, dfAttack, clusters, nearest):
        # trasformo i dataframe in array
        X_Normal = np.array(dfNormal.drop(['classification'], 1).astype(float))
        X_Attack = np.array(dfAttack.drop(['classification'], 1).astype(float))

        rowAttacks = np.size(X_Attack, 0)
        rowNormal = np.size(X_Normal, 0)

        print(rowAttacks)
        print(rowNormal)

        centerAtt = self.clusteringMiniBatchKMean(X_Attack, clusters, rowAttacks)
        print("AttDone")
        centerNorm = self.clusteringMiniBatchKMean(X_Normal, clusters, rowNormal)
        print("NormDone")

        matriceDistTrain = []
        matriceDistTest = []

        # ciclo sull'intero dataset per calcolare per ogni esempio, i centroide più vicini
        for i in range(len(X_Train)):
            feature1 =self.reshapeFeature(X_Train[i])
            dist_matrixN = pairwise_distances_argmin_min(feature1, centerNorm)
            if nearest == True:
                if dist_matrixN[1] == 0:
                    ind = dist_matrixN[0]
                    centerNorm[ind, :] = 0
                    dist_matrixN = pairwise_distances_argmin_min(feature1, centerNorm)
                    centerNorm[ind] = feature1

            dist_matrixA = pairwise_distances_argmin_min(feature1, centerAtt)
            if nearest == True:
                if dist_matrixA[1] == 0:
                    ind = dist_matrixA[0]
                    centerAtt[ind, :] = 0
                    dist_matrixA = pairwise_distances_argmin_min(feature1, centerAtt)
                    centerAtt[ind] = feature1

            row = [X_Train[i], centerNorm[dist_matrixN[0].item()], centerAtt[dist_matrixA[0].item()]]
            matriceDistTrain.append(row)


        for i in range(len(X_Test)):
            feature1 = self.reshapeFeature(X_Test[i])
            dist_matrixN = pairwise_distances_argmin_min(feature1, centerNorm)
            dist_matrixA = pairwise_distances_argmin_min(feature1, centerAtt)
            row = [X_Test[i], centerNorm[dist_matrixN[0].item()], centerAtt[dist_matrixA[0].item()]]
            matriceDistTest.append(row)

        return matriceDistTrain, matriceDistTest


    def distanceAll(self,X_Train, X_Test, dfNormal, dfAttack, clusters):
        # trasformo i dataframe in array
        X_Normal = np.array(dfNormal.drop(['classification'], 1).astype(float))
        X_Attack = np.array(dfAttack.drop(['classification'], 1).astype(float))

        rowAttacks=np.size(X_Attack,0)
        rowNormal=np.size(X_Normal,0)

        print(rowAttacks)
        print(rowNormal)

        centerAtt = self.clusteringMiniBatchKMean(X_Attack, clusters, rowAttacks)
        print("AttDone")
        centerNorm = self.clusteringMiniBatchKMean(X_Normal, clusters, rowNormal)
        print("NormDone")

        matriceDistTrain = []
        matriceDistTest = []

        # ciclo sull'intero dataset per calcolare per ogni esempio, i centroide più vicini
        for i in range(len(X_Train)):
            feature1 = self.reshapeFeature(X_Train[i])
            dist_matrixN = pairwise_distances_argmin_min(feature1, centerNorm)
            dist_matrixA = pairwise_distances_argmin_min(feature1, centerAtt)
            row = [X_Train[i], centerNorm[dist_matrixN[0].item()], centerAtt[dist_matrixA[0].item()]]
            matriceDistTrain.append(row)

        for i in range(len(X_Test)):
            feature1 = self.reshapeFeature(X_Test[i])
            dist_matrixN = pairwise_distances_argmin_min(feature1, centerNorm)
            dist_matrixA = pairwise_distances_argmin_min(feature1, centerAtt)
            row = [X_Test[i], centerNorm[dist_matrixN[0].item()], centerAtt[dist_matrixA[0].item()]]
            matriceDistTest.append(row)

        return matriceDistTrain, matriceDistTest


    def distanceNOCls(self, X_Train, X_Test, dfNormal, dfAttack ):
        X_Normal = np.array(dfNormal.drop(['classification'], 1).astype(float))
        X_Attack = np.array(dfAttack.drop(['classification'], 1).astype(float))
        matriceDistTrain = []
        matriceDistTest = []

        for i in range(len(X_Train)):
            feature1 = self.reshapeFeature(X_Train[i])
            dist_matrixN = pairwise_distances_argmin_min(feature1, X_Normal)
            dist_matrixA = pairwise_distances_argmin_min(feature1, X_Attack)
            row = [X_Train[i],  X_Normal[dist_matrixN[0].item()], X_Attack[dist_matrixA[0].item()]]
            matriceDistTrain.append(row)
        print("Training done")

        for i in range(len(X_Test)):
            feature1 = self.reshapeFeature(X_Test[i])
            dist_matrixN = pairwise_distances_argmin_min(feature1, X_Normal)
            dist_matrixA = pairwise_distances_argmin_min(feature1, X_Attack)
            row = [X_Test[i], X_Normal[dist_matrixN[0].item()], X_Attack[dist_matrixA[0].item()]]
            matriceDistTest.append(row)
        print("Test done")

        return matriceDistTrain, matriceDistTest




    def findNearest(self,df, dfNormal, dfAttack, train):
        trainNormal = train[train['classification'] == 1]
        trainAttack = train[train['classification'] == 0]
        df_NormalT = trainNormal.drop(['classification'], 1)
        df_AttackT = trainAttack.drop(['classification'], 1)
        X_NormalT = np.array(df_NormalT.drop(['classification'], 1).astype(float))
        X_AttackT = np.array(df_AttackT.drop(['classification'], 1).astype(float))

        dfNormal1 = dfNormal.drop(['classification'], 1)
        dfAttack1 = dfAttack.drop(['classification'], 1)
        X_Normal = np.array(dfNormal.drop(['classification'], 1).astype(float))
        X_Attack = np.array(dfAttack.drop(['classification'], 1).astype(float))

        matriceDist = []

        for i in range(len(X_Normal)):
            df2 = dfNormal1._slice(slice(i, i + 1))
            df2 = np.array(df2).astype(float)
            feature1 = self.reshapeFeature(df2[0])
            X_Normal[i, :] = 0

            dist_matrixN = pairwise_distances_argmin_min(feature1, X_NormalT)
            dist_matrixA = pairwise_distances_argmin_min(feature1, X_AttackT)
            row = [X_Normal[i], X_NormalT[dist_matrixN[0].item()], X_AttackT[dist_matrixA[0].item()]]
            matriceDist.append(row)
            X_Normal[i] = feature1

        for i in range(len(X_Attack)):
            df2 = dfAttack1._slice(slice(i, i + 1))
            df2 = np.array(df2).astype(float)
            feature1 = self.reshapeFeature(df2[0])
            X_Attack[i, :] = 0

            dist_matrixN = pairwise_distances_argmin_min(feature1, X_Normal)
            dist_matrixA = pairwise_distances_argmin_min(feature1, X_Attack)
            row = [X_Attack[i], X_Normal[dist_matrixN[0].item()], X_Attack[dist_matrixA[0].item()]]
            matriceDist.append(row)
            X_Attack[i] = feature1

        return matriceDist

    def findNearestTrain(self,df, dfNormal, dfAttack):
        dfNormal1 = dfNormal.drop(['classification'], 1)
        dfAttack1 = dfAttack.drop(['classification'], 1)
        X_Normal = np.array(dfNormal.drop(['classification'], 1).astype(float))
        X_Attack = np.array(dfAttack.drop(['classification'], 1).astype(float))

        matriceDist = []

        for i in range(len(X_Normal)):
            df2 = dfNormal1._slice(slice(i, i + 1))
            df2 = np.array(df2).astype(float)
            feature1 = self.reshapeFeature(df2[0])
            X_Normal[i, :] = 0

            dist_matrixN = pairwise_distances_argmin_min(feature1, X_Normal)
            dist_matrixA = pairwise_distances_argmin_min(feature1, X_Attack)
            row = [X_Normal[i], X_Normal[dist_matrixN[0].item()], X_Attack[dist_matrixA[0].item()]]
            matriceDist.append(row)
            X_Normal[i] = feature1

        for i in range(len(X_Attack)):
            df2 = dfAttack1._slice(slice(i, i + 1))
            df2 = np.array(df2).astype(float)
            feature1 = self.reshapeFeature(df2[0])
            X_Attack[i, :] = 0

            dist_matrixN = pairwise_distances_argmin_min(feature1, X_Normal)
            dist_matrixA = pairwise_distances_argmin_min(feature1, X_Attack)
            row = [X_Attack[i], X_Normal[dist_matrixN[0].item()], X_Attack[dist_matrixA[0].item()]]
            matriceDist.append(row)
            X_Attack[i] = feature1

        return matriceDist

    def findNearestTest(self,X_test, dfNormal, dfAttack):
        XNormal = np.array(dfNormal.drop(['classification'], 1).astype(float))
        XAttack = np.array(dfAttack.drop(['classification'], 1).astype(float))

        matriceDistTest = []

        for i in range(len(X_test)):
            feature1 = self.reshapeFeature(X_test[i])
            dist_matrixN = pairwise_distances_argmin_min(feature1, XNormal)
            dist_matrixA = pairwise_distances_argmin_min(feature1, XAttack)
            row = [X_test[i], XNormal[dist_matrixN[0].item()], XAttack[dist_matrixA[0].item()]]
            matriceDistTest.append(row)

        return matriceDistTest

