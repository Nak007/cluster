'''
Available methods are the followings:
[1] FactorRotation
[2] PrincipalComponents
[3] Cal_Bartlett
[4] Cal_KMOScore 
[5] Cal_PartialCorr 
[6] Cal_SMC

Authors: Danusorn Sitdhirasdr <danusorn.si@gmail.com>
versionadded:: 22-07-2021

'''
import pandas as pd, numpy as np, math
from scipy.stats import gaussian_kde
from scipy.ndimage.filters import gaussian_filter1d
from scipy import (stats, linalg)

import matplotlib.pylab as plt
from matplotlib.colors import ListedColormap
from matplotlib import cm
from itertools import product
import collections

__all__  = ["FactorRotation", 
            "PrincipalComponents", 
            "Cal_Bartlett", 
            "Cal_KMOScore", "KMO_table",
            "Cal_PartialCorr", 
            "Cal_SMC"]

class FactorRotation():
    
    '''
    FactorRotation provides statistical information towards 
    interpretation of factor loadings, specific variances, 
    and communalities from "factor-analyzer" [1].
   
    Atrributes
    ----------
    loadings : pd.DataFrame
        The correlation coefficient between variables and factors.

    communalities : pd.DataFrame
        The elements are squared loadings, which represent the 
        variance of each item that can be explainded by the 
        corresponding factor. 

    variances : pd.DataFrame
        Total Variance Explained of initial eigenvalues (PCA), 
        and after extraction. 

    common_variances : pd.DataFrame
        Correlation or variance between variables that can be 
        explained by factors. Communalities are on the diagonal. 
        Value closer to 1 suggests that factors explain more of 
        the variance of variables [1].

    unique_variances : pd.DataFrame
        Unique variance is comprised of specific variances 
        (diagonal) and residual variances (off-diagonal). These 
        variances can not be explained by factors [1]. 

    rmsr : float
        According to the model assumptions stating that specific 
        factors are uncorrelated with one another, cov(ϵi,ϵj) = 0 
        for i ≠ j, the off-diagonal elements from unique_variances 
        should be small towards zeros, which can be measured by
        Root-Mean-Square Residual (RMSR) [1].

    References
    ----------
    .. [1] https://factor-analyzer.readthedocs.io/en/latest/factor_
           analyzer.html#factor-analyzer-analyze-module
    .. [2] https://online.stat.psu.edu/stat505/book/export/html/691

    '''
    def __init__(self):
        pass
        
    def fit(self, estimator, features=None):
        
        '''
        Fit model.
        
        Parameters
        ----------
        estimator : estimator object.
            An estimator of "factor-analyzer" or other that implements 
            the similar interface [1].
        
        features : list of str, default=None
            A list of features. If None, features default to ["X1","X2", 
            ...,"Xn"] where n is the number of features according to 
            "loadings".
            
        Atrributes
        ----------
        loadings : pd.DataFrame
            The correlation coefficient between variables and factors.

        communalities : pd.DataFrame
            The elements are squared loadings, which represent the 
            variance of each item that can be explainded by the 
            corresponding factor. 

        variances : pd.DataFrame
            Total Variance Explained of initial eigenvalues (PCA), 
            and after extraction. 
            
        common_variances : pd.DataFrame
            Correlation or variance between variables that can be 
            explained by factors. Communalities are on the diagonal. 
            Value closer to 1 suggests that factors explain more of 
            the variance of variables [1].

        unique_variances : pd.DataFrame
            Unique variance is comprised of specific variances 
            (diagonal) and residual variances (off-diagonal). These 
            variances can not be explained by factors [1]. 

        rmsr : float
            According to the model assumptions stating that specific 
            factors are uncorrelated with one another, cov(ϵi,ϵj) = 0 
            for i ≠ j, the off-diagonal elements from unique_variances 
            should be small towards zeros, which can be measured by
            Root-Mean-Square Residual (RMSR) [1].
        
        References
        ----------
        .. [1] https://factor-analyzer.readthedocs.io/en/latest/factor_
               analyzer.html#factor-analyzer-analyze-module
        .. [2] https://online.stat.psu.edu/stat505/book/export/html/691
        
        '''
        # Attributes from estimator
        n_factors = estimator.n_factors
        loadings  = estimator.loadings_
        corr_     = estimator.corr_
        n_features= len(corr_)
 
        factors = [f"F{n}" for n in range(1, n_factors+1)]
        if features is not None: features = list(features)
        else: features = [f"X{n+1}" for n in range(n_features)]
    
        # Initial and extracted variances
        columns = ["Total", "% Variance", "Cumulative %"]
        columns = list(product(["Initial","Extraction"], columns))
        columns = pd.MultiIndex.from_tuples(columns)
        index   = pd.Index([f"F{n+1}" for n in range(n_features)])
        
        initial = estimator.get_eigenvalues()[0]  
        initvar = initial / n_features
        initial = np.vstack((initial, initvar, np.cumsum(initvar))).T
        extract = np.vstack((np.vstack(estimator.get_factor_variance()).T,
                             np.full((n_features-n_factors,3), [np.nan]*3)))
        self.variances = pd.DataFrame(np.hstack((initial, extract)), 
                                      columns=columns, index=index)
    
        # Communality (common variance), and 
        # Uniqueness (specific variance + error).
        columns =(list(product(["Extraction"], factors + 
                               ["Communality","Uniqueness"])))
        columns = pd.MultiIndex.from_tuples(columns)
    
        commu = (loadings**2).sum(axis=1, keepdims=True)
        data  = np.hstack((loadings**2, commu, 1-commu))
        data  = np.vstack((data, data.sum(0, keepdims=True)))
        self.communalities = pd.DataFrame(data, columns=columns,
                                          index=features + ["Total"])
                                          
        # Correlation and Cov(e(i),e(j)) given n_factors
        kwds = dict(columns=features, index=features)
        corr = loadings.dot(loadings.T)
        self.common_variances = pd.DataFrame(corr, **kwds)
        self.unique_variances = corr_ - self.common_variances
            
        # Calculate Root Mean Square Residual (RMSR)
        # res(i,j) = Cov(e(i),e(j)) = 0
        off_diag = ~np.diag(np.full(n_features, True))
        errors = sum((corr_-corr)[off_diag]**2)
        denom  = n_features*(n_features-1)
        self.rmsr = np.sqrt(errors/denom)
        self.loadings = pd.DataFrame(loadings, 
                                     columns=factors, 
                                     index=features)
                                     
        return self
    
    def plotting(self, value=None, ax=None, cmap=None, 
                 pcolor_kwds=None, anno_kwds=None, 
                 anno_format=None, tight_layout=True):
        
        '''
        Plot chart of eigenvectors or correlations.

        Parameters
        ----------
        value : {"loading", "common", "unique"}
            Data input to be used in the functon. If None, it 
            defaults to "loading"
            
        ax : Matplotlib axis object, default=None
            Predefined Matplotlib axis. If None, ax is created 
            with default figsize.

        cmap : str or Colormap, default=None
            A Colormap instance e.g. cm.get_cmap('Reds',20) or 
            registered colormap name. If None, it defaults to 
            "Blues". This overrides "pcolor_kwds".

        pcolor_kwds : keywords, default=None
            Keyword arguments to be passed to "ax.pcolor".
            
        anno_kwds : keywords, default=None
            Keyword arguments to be passed to "ax.annotate".
            
        anno_format : string formatter, default=None
            String formatters (function) for ax.annotate values. 
            If None, it defaults to "{:,.2f}".format.
            
        tight_layout : bool, default=True
            If True, it adjusts the padding between and around 
            subplots i.e. plt.tight_layout().

        Returns
        -------
        ax : Matplotlib axis object
        
        '''
        params = {"loading": ("loadings", 
                              r"Correlations, $\hat{L}=corr(X,PC)$"), 
                  "common" : ("common_variances", 
                              r"Correlations, $\hat{L}=corr(X,PC)$"), 
                  "unique" : ("unique_variances", 
                              r"Residual variances, $cov(e_{i},e_{j})$")}
        
        value = "loading" if value is None else value
        data, title = params[value]
        data = getattr(self, data)
        data = data.reindex(index=data.index[::-1])
        n_rows, n_cols = data.shape
 
        # Create matplotlib.axes if ax is None.
        if ax is None: 
            figsize=(n_cols, n_rows-.5)
            ax = plt.subplots(figsize=figsize)[1]
        
        # Matplotlib Colormap
        if cmap is None: cmap = 'Blues' 
        if isinstance(cmap, str): cmap = cm.get_cmap(cmap,50)
        
        abs_val = abs(data).values.ravel()
        kwds = dict(edgecolors='#4b4b4b', lw=1, alpha=0.8, 
                    vmin=min(abs_val), vmax=max(abs_val))
        kwds = (kwds if pcolor_kwds is None 
                else {**kwds, **pcolor_kwds})
        ax.pcolor(abs(data), **{**kwds,**{"cmap":cmap}})

        # Annotation.
        anno_xy = [[m+0.5,n+0.5] for n in range(n_rows) 
                   for m in range(n_cols)]
        anno_format = ("{:,.2f}".format if anno_format 
                       is None else anno_format)
        kwds = dict(xytext =(0,0), textcoords='offset points', 
                    va='center', ha='center', fontsize=12, 
                    fontweight='demibold', color="Black")
        kwds = (kwds if anno_kwds is None else {**kwds,**anno_kwds})
        
        for xy, v in zip(anno_xy, data.values.ravel()): 
            ax.annotate(anno_format(v), xy, **kwds)

        ax.tick_params(tick1On=False)
        ax.set_xticks(np.arange(0.5, n_cols))
        ax.set_xticklabels(data.columns)
        ax.set_yticks(np.arange(0.5,n_rows))
        ax.set_yticklabels(data.index)
        ax.set_title(title, fontsize=14)
        if tight_layout: plt.tight_layout()
            
        return ax
    
class PrincipalComponents:
    
    '''
    PrincipalComponents performs dimension reduction algorithm 
    so-called Principal Component Analysis (PCA) on the correlation 
    of X.
    
    Parameters
    ----------
    mineigval : float, default=1
        Minimum value of eigenvalues when choosing number of 
        prinicpal components. The algorithm chooses factor, whose 
        eigenvalue is more than mineigval. Only available when
        method is either "eigval" or None.

    minprop : float, default=0.8
        Minimum proportion of variation explained when choosing 
        number of prinicpal components. The algorithm select a
        group of factors (ranked from highest to lowest by 
        eigenvalues), whose sum of variation explained is greater 
        than or equal to minprop.
        
    method : {"eigval", "varprop"}, default=None
        If "eigval", mineigval is selected as a threshold, whereas 
        "varprop" uses minprop. If None, a maximum number of 
        factors is selected between two methods.
        
    Attributes
    ----------
    eigvals : ndarray of shape (n_components,)
        The variance that get explained by factors

    eigvecs : pd.DataFrame of shape (n_features, n_components)
        Eigenvectors (or factors) are vectors whose direction 
        remain unchanged when a linear transformation is applied. 
        They represent the directions of maximum variance. The 
        factors are sorted by eigvals.

    variances : pd.DataFrame
        Variance that can be explained by a given factor. Starting 
        from the first factor, each subsequent factor is obtained 
        from partialling out the previous factor. Therefore the 
        first factor explains the most variance, and the last factor 
        explains the least [1].

    communalities : pd.DataFrame
        The communality is the sum of the squared component loadings 
        up to the number of components that gets extracted [1].

    References
    ----------
    .. [1] https://stats.idre.ucla.edu/spss/seminars/efa-spss/
   
    '''
    def __init__(self, minprop=0.8, mineigval=1.0, method=None):
    
        self.minprop = minprop
        self.mineigval = mineigval
        self.method = method
    
    def fit(self, X):
        
        '''
        Fit PCF model.
        
        Parameters
        ----------
        X : pd.DataFrame, of shape (n_samples, n_features)
            Sample data.
        
        Atrributes
        ----------
        eigvals : ndarray of shape (n_components,)
            The variance that get explained by factors
        
        eigvecs : pd.DataFrame of shape (n_features, n_components)
            Eigenvectors (or factors) are vectors whose direction 
            remain unchanged when a linear transformation is applied. 
            They represent the directions of maximum variance. The 
            factors are sorted by eigvals.
            
        variances : pd.DataFrame
            Variance that can be explained by a given factor. Starting 
            from the first factor, each subsequent factor is obtained 
            from partialling out the previous factor. Therefore the 
            first factor explains the most variance, and the last factor 
            explains the least [1].
            
        communalities : pd.DataFrame
            The communality is the sum of the squared component loadings 
            up to the number of components that gets extracted [1].
            
        References
        ----------
        .. [1] https://stats.idre.ucla.edu/spss/seminars/efa-spss/
 
        '''
        # Create PC columns
        width = int(np.ceil(np.log(X.shape[1])/np.log(10)))
        self.components = ["PC{}".format(str(n).zfill(width)) 
                           for n in range(1,X.shape[1]+1)]
        
        # Standardize X
        self.mean = np.mean(X.values, axis=0)
        self.stdv = np.std(X.values, axis=0)
        std_X = ((X-self.mean)/self.stdv).copy()
        
        # Correlation matrix
        corr = pd.DataFrame(std_X).corr().values
        
        # Eigenvalues, Eigenvectors, and loadings
        eigvals, eigvecs = np.linalg.eigh(corr)
        indices = np.argsort(eigvals)[::-1]
        self.eigvals = eigvals[indices].real
        self.eigvecs = pd.DataFrame(eigvecs[:,indices].real, 
                                      columns=self.components, 
                                      index=X.columns)
        loadings = self.eigvecs*np.sqrt(self.eigvals)
        
        # Variance explained
        varprops = self.eigvals/self.eigvals.sum()
        cumprops = np.cumsum(varprops)
        
        # factors
        data = np.vstack((self.components, self.eigvals, 
                          varprops, cumprops))
        columns = ["Factor", "Eigenvalues", "% Variance", "Cumulative %"]
        self.variances = (pd.DataFrame(data.T, columns=columns)
                          .set_index("Factor").astype("float64"))
    
        # Recommended number of factors
        n_minprop = np.argmax((cumprops>=self.minprop))+1
        n_maxeigval = (eigvals>=self.mineigval).sum()

        if self.method=="eigval": n_factors = n_maxeigval
        elif self.method=="varprop": n_factors = n_minprop
        else: n_factors = max([n_minprop, n_maxeigval, 1])
        self.n_factors = n_factors
        
        # Communalities
        columns = list(product(["Extraction"], 
                               self.components[:n_factors] +
                               ["Communality"]))
        columns = pd.MultiIndex.from_tuples(columns)
        variances = (loadings.values**2)[:,:n_factors]
        communalities = variances.sum(axis=1, keepdims=True)
        data  = np.hstack((variances, communalities))
        data  = np.vstack((data, data.sum(0, keepdims=True)))
        self.communalities = pd.DataFrame(data, 
                                          columns=columns, 
                                          index=list(X)+["Total"])
        return self

    def transform(self, X, n_factors=None):

        '''
        Apply the dimensionality reduction on X.
        
        Parameters
        ----------
        X : pd.DataFrame, of shape (n_samples, n_features)
            Sample data.
        
        n_factors : int, default=None
            Number of factors. If None, it is selected according 
            to "method".
            
        Returns
        -------
        PC : pd.DataFrame, of shape (n_samples, n_comps)
            Transformed X.
        
        '''
        std_X = (X-self.mean)/self.stdv
        n_factors = self.n_factors if n_factors is None else max(n_factors,1)
        PC = np.dot(std_X, self.eigvecs.values[:,:n_factors])
        return pd.DataFrame(PC, columns=self.components[:n_factors])
    
    def fit_transform(self, X, n_factors=None):
        
        '''
        Fit X and apply the dimensionality reduction on X.
        
        Parameters
        ----------
        X : pd.DataFrame, of shape (n_samples, n_features)
            Sample data.
        
        n_factors : int, default=None
            Number of factors. If None, it is selected according 
            to "method".
            
        Returns
        -------
        PC : pd.DataFrame, of shape (n_samples, n_comps)
            Transformed X.
        '''
        self.fit(X)
        return self.transform(X, n_factors)
    
def Cal_Bartlett(X):
    
    '''
    Bartlett's Sphericity tests the hypothesis that the 
    correlation matrix is equal to the identity matrix, 
    which would indicate that your variables are unrelated 
    and therefore unsuitable for structure detection [1].
    
        H0: The correlation matrix is equal to I 
        H1: The correlation matrix is not equal to I
        
    References
    ----------
    .. [1] https://www.ibm.com/docs/en/spss-statistics/
           23.0.0?topic=detection-kmo-bartletts-test 
    
    Parameters
    ----------
    X : pd.DataFrame, of shape (n_samples, n_features)
        Sample data.

    Returns
    -------
    BartlettTest : collections.namedtuple
        Bartlett's Sphericity test result.
        
    '''
    corr = np.corrcoef(X.T)
    R = np.linalg.det(corr)
    p = corr.shape[0]
    df= p*(p-1)/2
    chisq = -((len(X)-1)-(2*p+5)/6)*np.log(R)
    p_value = 1-stats.chi2.cdf(chisq, df=df)
    critval= stats.chi2.ppf(1-0.05, df)
    
    keys = ["chisq", "df", "pvalue"]
    BTest = collections.namedtuple('BartlettTest', keys)
    BTest = BTest(*(chisq, df, p_value))

    return BTest

def Cal_KMOScore(X):
    
    '''
    The Kaiser-Meyer-Olkin Measure of Sampling Adequacy is a 
    statistic that indicates the proportion of variance in 
    your variables that might be caused by underlying factors. 
    High values (close to 1.0) generally indicate that a factor 
    analysis may be useful with your data. If the value is less 
    than 0.50, the results of the factor analysis probably 
    won't be very useful.
    
    References
    ----------
    .. [1] https://www.ibm.com/docs/en/spss-statistics/
           23.0.0?topic=detection-kmo-bartletts-test
    .. [2] https://factor-analyzer.readthedocs.io/en/latest/
           _modules/factor_analyzer/factor_analyzer.html

    Parameters
    ----------
    X : pd.DataFrame, of shape (n_samples, n_features)
        Sample data.

    Returns
    -------
    MSATest : collections.namedtuple
        Measure of Sampling Adequacy (MSA).
        
    KMOTest : collections.namedtuple
        Kaiser-Meyer-Olkin (KMO).
        
    '''
    # Pair-wise correlations
    diag = np.identity(X.shape[1])
    corr = (X.corr().values**2) - diag
    
    # Partial correlations
    pcorr = (Cal_PartialCorr(X).values**2) - diag

    # Measure of Sampling Adequacy (MSA)
    pcorr_sum = np.sum(pcorr, axis=0)
    corr_sum  = np.sum(corr, axis=0)
    msa_score = corr_sum / (corr_sum + pcorr_sum)
    
    keys = ["Score", "Corr", "PartialCorr"]
    MSATest = collections.namedtuple('MSATest', keys)
    MSATest = MSATest(*(msa_score, corr_sum, pcorr_sum))

    # Kaiser-Meyer-Olkin (KMO)
    kmo_score = np.sum(corr) / (np.sum(corr) + np.sum(pcorr))
    KMOTest = collections.namedtuple('KMOTest', keys)
    KMOTest = KMOTest(*(kmo_score, np.sum(corr), np.sum(pcorr)))
    
    return MSATest, KMOTest

def Cal_PartialCorr(X):
    
    '''
    Partial correlation coefficients describes the linear 
    relationship between two variables while controlling 
    for the effects of one or more additional variables [1].
    
    If we want to find partial correlation of X, and Y while
    controlling Z. we regress variable X on variable Z, then 
    subtract X' from X, we have a residual eX. This eX will 
    be uncorrelated with Z, so any correlation X shares with 
    another variable Y cannot be due to Z. This method also 
    applies to Y in order to compute eY. The correlation 
    between the two sets of residuals, corr(e(X), e(Y)) is 
    called a partial correlation [2]. 
    
    Parameters
    ----------
    X : pd.DataFrame, of shape (n_samples, n_features)
        Sample data.
            
    References
    ----------
    .. [1] https://www.ibm.com/docs/en/spss-statistics/
           24.0.0?topic=option-partial-correlations
    .. [2] http://faculty.cas.usf.edu/mbrannick/regression/
           Partial.html
    
    Returns
    -------
    pcorr : pd.DataFrame of shape (n_features, n_features)
        Partial correlations.
    
    '''
    n_features = X.shape[1]
    X0 = np.array(X).copy()
    pcorr = np.zeros((n_features,)*2)
    
    for i,j in product(*((range(n_features),)*2)):
        if j-i > 0:
            resids = []
            # Controlled variables
            index = np.isin(range(n_features),[i,j])
            Xs = np.hstack((np.ones((X0.shape[0],1)), X0[:,~index]))
            
            for Z in (X0[:,[i]], X0[:,[j]]):
                
                # Determine betas (INV(X'X)X'Y) and residuals
                betas = np.linalg.inv(Xs.T.dot(Xs)).dot(Xs.T).dot(Z)
                resids.append(Z.ravel() - Xs.dot(betas).ravel())
                
            # Partial correlation between xi, and xj
            pr = np.corrcoef(np.array(resids))[0,1]
            pcorr[i,j] = pr
        
    pcorr = pcorr + pcorr.T 
    return pd.DataFrame(pcorr + np.identity(n_features), 
                        columns=X.columns, 
                        index=X.columns)

def Cal_SMC(X):
    
    '''
    Calculate the squared multiple correlations.
    This is equivalent to regressing each variable
    on all others and calculating the r2 values.
    
    Parameters
    ----------
    X : pd.DataFrame, of shape (n_samples, n_features)
        Sample data.

    Returns
    -------
    smc : numpy array
        The squared multiple correlations matrix.
        
    '''  
    # smc = 1-1/np.diag(np.linalg.inv(np.corrcoef(X.T)))
    X0, smc = np.array(X), []
    for i in range(X0.shape[1]):
        
        # Controlled variables
        index = np.isin(range(X0.shape[1]),[i])
        Xs = np.hstack((np.ones((X0.shape[0],1)), X0[:,~index]))
        y  = X0[:,[i]]
        
        # Determine betas (INV(X'X)X'Y) and residuals
        betas = np.linalg.inv(Xs.T.dot(Xs)).dot(Xs.T).dot(y)
        smc.append(np.var(Xs.dot(betas))/np.var(y))
    
    return pd.DataFrame(smc, columns=["SMC"], 
                        index=X.columns)

def KMO_table():
    standard = [["0.0 $\geq$ KMO $>$ 0.5", "unacceptable"],
                ["0.5 $\geq$ KMO $>$ 0.6", "miserable"],
                ["0.6 $\geq$ KMO $>$ 0.7", "mediocre"],
                ["0.7 $\geq$ KMO $>$ 0.8", "middling"],
                ["0.8 $\geq$ KMO $>$ 0.9", "meritorious"],
                ["0.9 $\geq$ KMO $\geq$ 1.0", "marvelous"]]
    return pd.DataFrame(standard, columns=["KMO","Suitability"]) 