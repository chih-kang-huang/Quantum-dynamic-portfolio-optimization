import numpy as np
import yfinance as yf

class PortfolioModel:
    """
    A portfolio model container for storing mean vectors, covariance matrices, and parameters.
    """

    def __init__(self, mu_list, Sigma_list, c_list, lam, eta_list, A_list, Nq):
        """
        Initialize the portfolio model.

        Parameters
        ----------
        mu_list : list of np.ndarray
            List of mean vectors (shape d,).
        Sigma_list : list of np.ndarray
            List of covariance matrices (shape d x d).
        c_list : list of float
            Coefficients associated with each asset/component.
        lam : float
            Regularization or scaling parameter.
        eta_list : list of float
            Learning or perturbation parameters.
        A_list : list of float
            Additional parameters, one per asset/component.
        Nq : int
            Quantization level (e.g. bits per weight).
        """
        self.mu_list = mu_list
        self.Sigma_list = Sigma_list
        self.c_list = c_list
        self.lam = lam
        self.eta_list = eta_list
        self.A_list = A_list
        self.Nq = Nq
        self.N = len(mu_list[0]) if mu_list else 0
        self.Nt = len(mu_list)

        self._check_consistency()

    def _check_consistency(self):
        """Check dimensions and consistency of parameters."""
        n = len(self.mu_list)

        if not (len(self.Sigma_list) == len(self.c_list) == len(self.A_list) == n):
            raise ValueError("mu_list, Sigma_list, c_list, and A_list must all have the same length.")

        d = self.mu_list[0].shape[0]
        for mu in self.mu_list:
            if mu.shape[0] != d:
                raise ValueError("All mean vectors in mu_list must have the same dimension.")

        for Sigma in self.Sigma_list:
            if Sigma.shape != (d, d):
                raise ValueError("Each covariance matrix must be square of shape (d, d).")

    def summary(self):
        """Print a summary of the portfolio model parameters."""
        print("Portfolio Model Summary:")
        print(f"  Number of assets      : {self.N}")
        print(f"  Number of time steps  : {self.Nt}")
        print(f"  λ (lambda)            : {self.lam}")
        print(f"  Nq (binary encoding)  : {self.Nq}")
        print(f"  mu_list               : {self.mu_list}")
        print(f"  Sigma_list            : {self.Sigma_list}")
        print(f"  c_list                : {self.c_list}")
        print(f"  eta_list              : {self.eta_list}")
        print(f"  A_list                : {self.A_list}")

    def short_summary(self):
        """Print a summary of the portfolio model parameters."""
        print("Portfolio Model Summary:")
        print(f"  Number of assets      : {self.N}")
        print(f"  Number of time steps  : {self.Nt}")
        print(f"  λ (lambda)            : {self.lam}")
        print(f"  Nq (binary encoding)  : {self.Nq}")
        # print(f"  mu_list               : {self.mu_list[:2]} ... {self.mu_list[-1:]}")
        # print(f"  Sigma_list            : {self.Sigma_list[:2]} ... {self.Sigma_list[-1:]}")
        print(f"  c_list                : {self.c_list}")
        print(f"  eta_list              : {self.eta_list}")
        print(f"  A_list                : {self.A_list}")

    def decode(self, bits):
        Delta = 1.0/(2**(self.Nq+1)-1)  # 1/15
        x = np.zeros((self.Nt,self.N))
        stride = self.Nq+1  # 4
        for t in range(self.Nt):
            for i in range(self.Nt):
                seg = bits[i*stride:(i+1)*stride]
                x[t][i] = Delta*sum((1<<q)*int(seg[q]) for q in range(stride))
        return x

    def decode_solution(self, sample2):
        Delta = 1.0/(2**(self.Nq+1)-1)  # 1/15
        x = np.zeros((self.Nt,self.N))
        stride = self.Nq+1  # 4
        for (label, t, i, q), bit in sample.items():
            if bit == 1:
                x[t][i] += Delta * (2**q)
        return x

    def show_decode(self, x, precision=3, tickers=None):    
        Nt, N = x.shape
        assert Nt == self.Nt and N == self.N
    
        header = [" "] + [f"asset{j+1}" for j in range(N)]
        row_format = "{:>10}" * (N+1)
    
        print(row_format.format(*header))
    
        # Rows
        for i in range(Nt):
            row = [f"t={i}"] + [f"{x[i, j]:.{precision}f}" for j in range(N)]
            print(row_format.format(*row))




# -----------------------------
# Utility function for toy model
# -----------------------------
def create_toy_example():
    """Create and return a toy PortfolioModel instance."""
    mu_list = [
        np.array([0.1, 0.05]),
        np.array([0.08, 0.04])
    ]
    Sigma_list = [
        np.array([[0.02, 0.01], [0.01, 0.02]]),
        np.array([[0.03, 0.01], [0.01, 0.025]])
    ]
    c_list = [1.0, 0.5]
    lam = 1.0
    eta_list = [0.1]
    A_list = [5.0, 5.0]
    Nq = 1

    return PortfolioModel(mu_list, Sigma_list, c_list, lam, eta_list, A_list, Nq)

# ---------------------------------------
# Utility function: finance data example
# ---------------------------------------
def create_finance_example(window: int = 60, tickers=None):
    """
    Create a PortfolioModel using rolling mean and covariance estimates from financial data.

    Parameters
    ----------
    window : int
        Rolling window size in trading days (default 60 ≈ 3 months).
    tickers : list of str
        List of stock tickers. Defaults to 10 large-cap US stocks.

    Returns
    -------
    PortfolioModel
    """
    if tickers is None:
        tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN",
                   "NVDA", "GS", "MS", "KO", "NKE"]

    # Download daily adjusted close prices
    data = yf.download(tickers, start="2025-04-01", end="2025-10-03", auto_adjust=True)["Close"]

    # Compute log returns
    returns = np.log(data / data.shift(1)).dropna()

    # Rolling mean & covariance
    rolling_mean = returns.rolling(window).mean()
    rolling_cov = returns.rolling(window).cov()

    # Use last two days for illustration
    today = returns.index[-1]
    yesterday = returns.index[-2]

    mu_today = rolling_mean.loc[today].values
    cov_today = rolling_cov.loc[today].values

    mu_yesterday = rolling_mean.loc[yesterday].values
    cov_yesterday = rolling_cov.loc[yesterday].values

    mu_list_finance = [np.array(mu_yesterday), np.array(mu_today)]
    Sigma_list_finance = [np.array(cov_yesterday), np.array(cov_today)]

    # Parameters (example values)
    c_list = [1.0, 0.5]
    lam = 1.0
    eta_list = [0.1]
    A_list = [5.0, 5.0]
    Nq = 1

    return PortfolioModel(mu_list_finance, Sigma_list_finance, c_list, lam, eta_list, A_list, Nq)

if __name__ == "__main__":
    mu_list = [np.array([0.1, 0.05]), np.array([0.08, 0.04])]
    Sigma_list = [
        np.array([[0.02, 0.01], [0.01, 0.02]]), np.array([[0.03, 0.01], [0.01, 0.025]])
    ]
    c_list = [1.0, 0.5]
    lam = 1.0
    eta_list = [0.1]
    A_list = [5.0, 5.0]
    Nq = 1

    portfolio = PortfolioModel(mu_list, Sigma_list, c_list, lam, eta_list, A_list, Nq)
    portfolio.summary()
