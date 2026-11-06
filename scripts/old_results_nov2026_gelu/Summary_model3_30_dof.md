
======================================================================
 SUMMARY REPORT: MODEL3_30_DOF
 Target Evaluation Slack: 0.0
 Valid Boundary Points Analyzed: 2464
======================================================================
-> Global Minimum Pf : 3.163e-02 (Found during opt at slack 0.025)
-> Global Maximum Pf : 5.789e-02 (Found during opt at slack 0.0)
-> Magnitude Diff    : 0.26 orders of magnitude

[Dominant Variables - True Min Pf Allocation]
[0.     1.     0.     0.     1.     0.     0.     0.     1.     0.
 0.     1.     0.     0.     0.     0.     0.     1.     0.     1.
 0.     0.     1.     1.     1.     1.     0.2816 1.     1.     1.    ]

[Dominant Variables - True Max Pf Allocation]
[1.     1.     1.     1.     1.     1.     1.     1.     1.     1.
 1.     1.     1.     1.     1.     1.     1.     1.     1.     1.
 1.     1.     1.     1.     1.     1.     1.     0.     1.     0.4405]
======================================================================

### LaTeX Matrices
```latex
\begin{equation}
    \mathbf{x}_{\min} = \begin{bmatrix}
    \textcolor{green}{0.    } & \textcolor{red}{1.    } & \textcolor{green}{0.021 } & \textcolor{green}{0.0103} \\
    \textcolor{red}{1.    } & \textcolor{green}{0.0213} & \textcolor{green}{0.0161} & \textcolor{green}{0.0146} \\
    \textcolor{red}{1.    } & \textcolor{green}{0.    } & \textcolor{green}{0.0041} & \textcolor{red}{1.    } \\
    \textcolor{green}{0.0035} & \textcolor{green}{0.0093} & \textcolor{green}{0.    } & \textcolor{green}{0.    } \\
    \textcolor{green}{0.    } & \textcolor{red}{1.    } & \textcolor{green}{0.    } & \textcolor{red}{1.    } \\
    \textcolor{green}{0.0004} & \textcolor{green}{0.    } & \textcolor{red}{1.    } & \textcolor{red}{0.9929} \\
    \textcolor{red}{1.    } & \textcolor{red}{1.    } & \textcolor{black}{0.2816} & \textcolor{red}{1.    } \\
    \textcolor{red}{1.    } & \textcolor{red}{1.    }
    \end{bmatrix}^T
\label{eq:\min_standards}
\end{equation}

\begin{equation}
    \mathbf{x}_{\max} = \begin{bmatrix}
    \textcolor{red}{1.    } & \textcolor{red}{1.    } & \textcolor{red}{1.    } & \textcolor{red}{1.    } \\
    \textcolor{red}{1.    } & \textcolor{red}{1.    } & \textcolor{red}{1.    } & \textcolor{red}{1.    } \\
    \textcolor{red}{1.    } & \textcolor{red}{1.    } & \textcolor{red}{1.    } & \textcolor{red}{1.    } \\
    \textcolor{red}{1.    } & \textcolor{red}{1.    } & \textcolor{red}{1.    } & \textcolor{red}{1.    } \\
    \textcolor{red}{1.    } & \textcolor{red}{1.    } & \textcolor{red}{1.    } & \textcolor{red}{1.    } \\
    \textcolor{red}{1.    } & \textcolor{red}{1.    } & \textcolor{red}{1.    } & \textcolor{red}{1.    } \\
    \textcolor{red}{1.    } & \textcolor{red}{1.    } & \textcolor{red}{1.    } & \textcolor{green}{0.    } \\
    \textcolor{red}{1.    } & \textcolor{black}{0.4405}
    \end{bmatrix}^T
\label{eq:\max_standards}
\end{equation}
```
