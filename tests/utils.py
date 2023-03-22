import numpy as np 

def gen_lp_problem(c: np.array, k: float, u: np.array, G: np.array):
    # http://iiif.library.cmu.edu/file/Cooper_box00011_fld00008_bdl0001_doc0001/Cooper_box00011_fld00008_bdl0001_doc0001.pdf
    num_vars = c.shape[0]
    G_num_rows, G_num_cols = G.shape
    num_slack_vars = num_vars
    # left_inv_G = np.linalg.lstsq(G.T @ G, G.T)
    
    c = np.hstack([c, np.zeros(num_slack_vars)])
    A_top = np.hstack([np.ones((1, num_vars)), np.zeros((1, num_vars))])
    # G = np.eye(num_vars)
    A_bottom = np.hstack([G, G])
    A = np.vstack([A_top, A_bottom])

    b = np.hstack([np.array(k), (G @ u).flatten()])

    soln = np.zeros(num_vars+num_slack_vars)
    for i in range(num_vars):
        soln[i] += min(k - soln.sum(), u[i])

    # return {"c" : c, "A": A, "b": b, "soln": soln}
    return (c, A, b, np.arange(num_vars))


if __name__ == "__main__":
    import copy 
    n = 2
    c = np.ones(n)
    k = copy.copy(n)
    u = 1 * np.ones(n)
    G = np.array(
        [
            [1, 0],
            [0, 1], 
            
        ]
    )
    res = gen_lp_problem(c, k, u, G)
    print()