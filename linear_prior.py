import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.special import expit as sigmoid
from notears import helper
from datetime import datetime
from pytz import timezone
from notears import PriorKnowledge


def notears_linear_prior(X, lambda1, loss_type, max_iter=100, h_tol=1e-8, rho_max=1e+16, w_threshold=0.3, prior_knowledge = None, prior_k_weight = 1):
    """Solve min_W L(W; X) + lambda1 ‖W‖_1 s.t. h(W) = 0 using augmented Lagrangian.

    Args:
        X (np.ndarray): [n, d] sample matrix
        lambda1 (float): l1 penalty parameter
        loss_type (str): l2, logistic, poisson
        max_iter (int): max num of dual ascent steps
        h_tol (float): exit if |h(w_est)| <= htol
        rho_max (float): exit if rho >= rho_max
        w_threshold (float): drop edge if |weight| < threshold

    Returns:
        W_est (np.ndarray): [d, d] estimated DAG
    """
    def _loss(W):
        """Evaluate value and gradient of loss."""
        M = X @ W
        if loss_type == 'l2':
            R = X - M
            loss = 0.5 / X.shape[0] * (R ** 2).sum()
            G_loss = - 1.0 / X.shape[0] * X.T @ R
            if prior_knowledge is not None:
                dataset = prior_knowledge.get_dataset()
                prior_knowledge_matrix = prior_knowledge.get_prior_knowledge()
                for dataset_name in prior_knowledge_matrix:
                    if dataset_name == dataset:
                        for model in prior_knowledge_matrix[dataset_name]:
                            pr_g = prior_knowledge_matrix[dataset_name][model]
                            loss += 0.5 * prior_k_weight * prior_knowledge.LLM_weights[model] / X.shape[1] * ((pr_g - W) ** 2).sum()
                            G_loss += - 1.0 / X.shape[0] * prior_k_weight * (pr_g - W) * prior_knowledge.LLM_weights[model]

        elif loss_type == 'logistic':
            loss = 1.0 / X.shape[0] * (np.logaddexp(0, M) - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (sigmoid(M) - X)
        elif loss_type == 'poisson':
            S = np.exp(M)
            loss = 1.0 / X.shape[0] * (S - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (S - X)
        else:
            raise ValueError('unknown loss type')
        return loss, G_loss

    def _h(W):
        """Evaluate value and gradient of acyclicity constraint."""
        E = slin.expm(W * W)  # (Zheng et al. 2018)
        h = np.trace(E) - d
        #     # A different formulation, slightly faster at the cost of numerical stability
        #     M = np.eye(d) + W * W / d  # (Yu et al. 2019)
        #     E = np.linalg.matrix_power(M, d - 1)
        #     h = (E.T * M).sum() - d

        # if np.isnan(E).any() or np.isinf(E).any():
        #     print(W * W)
        #     print("Nan value in E")
        G_h = E.T * W * 2
        return h, G_h

    def _adj(w):
        """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
        return (w[:d * d] - w[d * d:]).reshape([d, d])

    def _func(w):
        """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
        W = _adj(w)
        loss, G_loss = _loss(W)
        h, G_h = _h(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
        G_smooth = G_loss + (rho * h + alpha) * G_h
        g_obj = np.concatenate((G_smooth + lambda1, - G_smooth + lambda1), axis=None)
        return obj, g_obj

    n, d = X.shape
    w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf  # double w_est into (w_pos, w_neg)
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
    if loss_type == 'l2':
        X = X - np.mean(X, axis=0, keepdims=True)
    for _ in range(max_iter):
        w_new, h_new = None, None
        while rho < rho_max:
            sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
            w_new = sol.x
            h_new, _ = _h(_adj(w_new))
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break
    W_est = _adj(w_est)
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est

def calculate_pure_score(W, X, rho = 1.0, lambda1 = 0.1):

    def _h(W, d):
        """Evaluate value and gradient of acyclicity constraint."""
        E = slin.expm(W * W)  # (Zheng et al. 2018)
        h = np.trace(E) - d
        #     # A different formulation, slightly faster at the cost of numerical stability
        #     M = np.eye(d) + W * W / d  # (Yu et al. 2019)
        #     E = np.linalg.matrix_power(M, d - 1)
        #     h = (E.T * M).sum() - d

        # if np.isnan(E).any() or np.isinf(E).any():
        #     print(W * W)
        #     print("Nan value in E")
        G_h = E.T * W * 2
        return h, G_h
    

    X = X - np.mean(X, axis=0, keepdims=True)
    M = X @ W
    R = X - M
    loss = 0.5 / X.shape[0] * (R ** 2).sum()
    h = _h(W, X.shape[1])[0]

    return loss + 0.5 * rho * h * h + lambda1 * W.sum()



if __name__ == '__main__':
    from notears import utils
    from linear import notears_linear
    import logging
    # utils.set_random_seed(1)

    # n, d, s0, graph_type, sem_type = 100, 20, 20, 'ER', 'gauss'
    # B_true = utils.simulate_dag(d, s0, graph_type)
    # W_true = utils.simulate_parameter(B_true)
    # np.savetxt('W_true.csv', W_true, delimiter=',')

    # X = utils.simulate_linear_sem(W_true, n, sem_type)
    # np.savetxt('X.csv', X, delimiter=',')
    logging.basicConfig(filename="result_log.log", level=logging.INFO)

    datasets = ['LUCAS', 'Asia', 'SACHS', 'Survey', 'Earthquake', 'Child', 'Alarm']

    for dataset in datasets:

        datapath, sol_path, plot_dir = helper.generate_data_path(dataset)
        prior_knowledge = PriorKnowledge(dataset, true_graph=False, LLMs = ['GPT4', 'Gemini', 'GPT3'])
        
        X = np.load(datapath).astype(np.float32) * 10
        B_true = np.load(sol_path).astype(np.float32)

        prior_knowledge.calculate_LLMs_weight(X)

        # Run original NOTEARS
        W_est = notears_linear(X, lambda1=0.1, loss_type='l2')
        assert utils.is_dag(W_est)


        # calculate accuracy
        acc = utils.count_accuracy(B_true, W_est != 0)
        score = calculate_pure_score(W_est, X)
        logging.info(f"---------------------------------------Starting {dataset}...")
        logging.info(f"No Prior Knowledge Accuracy: {acc}, Score: {score}")
        helper.plot_result(W_est != 0, B_true, plot_dir, dataset, 'notears_linear', acc)
        
        
        min_score = np.inf
        min_weight = None
        min_acc = None

        for weight in np.linspace(0.1, 10, 20):
            W_est_prior = notears_linear_prior(X, lambda1=0.1, loss_type='l2', prior_knowledge=prior_knowledge, prior_k_weight=weight, max_iter=100, w_threshold=0.3)
            assert utils.is_dag(W_est_prior)

            np.savetxt(f"{plot_dir}/W_est_prior_{datetime.now(timezone('Australia/Sydney')).strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3]}.csv", W_est_prior, delimiter=',')
            
            score = calculate_pure_score(W_est_prior != 0, X)
            # calculate accuracy
            acc_prior = utils.count_accuracy(B_true, W_est_prior != 0)

            if score < min_score:
                min_score = score
                min_weight = weight
                min_acc = utils.count_accuracy(B_true, W_est_prior != 0)

            logging.info(f"Prior Knowledge Accuracy with weight {weight}: {acc_prior}, Score: {score}")

            helper.plot_result(W_est_prior != 0, B_true, plot_dir, dataset, 'notears_linear_prior', acc_prior)
        
        for weight in np.linspace(10, 100, 20):
            W_est_prior = notears_linear_prior(X, lambda1=0.1, loss_type='l2', prior_knowledge=prior_knowledge, prior_k_weight=weight, max_iter=100, w_threshold=0.3)
            assert utils.is_dag(W_est_prior)

            np.savetxt(f"{plot_dir}/W_est_prior_{datetime.now(timezone('Australia/Sydney')).strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3]}.csv", W_est_prior, delimiter=',')
            
            # calculate accuracy
            score = calculate_pure_score(W_est_prior, X)
            acc_prior = utils.count_accuracy(B_true, W_est_prior != 0)

            if score < min_score:
                min_score = score
                min_weight = weight
                min_acc = utils.count_accuracy(B_true, W_est_prior != 0)

            logging.info(f"Prior Knowledge Accuracy with weight {weight}: {acc_prior}, Score: {score}")

            helper.plot_result(W_est_prior != 0, B_true, plot_dir, dataset, 'notears_linear_prior', acc_prior)
        

        
        logging.info(f"Finished {dataset}...---------------------------------")
        logging.info(f"Best Weight: {min_weight}, Best Accuracy: {min_acc}, Best Score: {min_score}")
        logging.info("------------------------------------------------------")
        logging.info("------------------------------------------------------")
        logging.info("------------------------------------------------------")
        
