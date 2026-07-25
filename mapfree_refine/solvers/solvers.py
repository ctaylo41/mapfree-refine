import numpy as np
from scipy.spatial.transform import RigidTransform as Tf
from scipy.spatial.transform import Rotation as R
import cv2

class Solvers():
    
    @staticmethod
    def weighted_procrustes(pts3d_ref, pts3d_query, weights, estimate_scale=False):

        weights = weights / weights.sum()

        pts3d_ref_mu = np.sum(weights[:, None] * pts3d_ref, axis=0)
        pts3d_query_mu = np.sum(weights[:, None] * pts3d_query, axis=0)

        pts3d_ref_centered = pts3d_ref - pts3d_ref_mu
        pts3d_query_centered = pts3d_query - pts3d_query_mu

        D_w = np.diag(weights)
        H = pts3d_ref_centered.T @ D_w @ pts3d_query_centered
        U, S, V_t = np.linalg.svd(H)

        Rot = V_t.T @ U.T

        det = np.linalg.det(Rot)
        d = np.array([1,1,np.sign(det)])

        Rot = V_t.T @ np.diag(d) @ U.T


        # Weighted variance
        pts3d_ref_var = np.sum(weights * np.sum(pts3d_ref_centered**2,axis=1))
        if estimate_scale:
            c = (S * d).sum() / pts3d_ref_var
        else:
            c = 1.0
        t = pts3d_query_mu - c * (Rot @ pts3d_ref_mu)



        return Tf.from_components(t,R.from_matrix(Rot))

    @staticmethod
    def pnp(pts3d_ref, pts2d_query, K_query, reprojection_threshold=3.0, iters=1000, conf=0.999):

        success, rvec, tvec, inliers = cv2.solvePnPRansac(objectPoints = pts3d_ref.astype(np.float64),
                                                        imagePoints = pts2d_query.astype(np.float64),
                                                        cameraMatrix = K_query,
                                                        distCoeffs = None,
                                                        reprojectionError=reprojection_threshold,
                                                        iterationsCount=iters,
                                                        confidence=conf)

        if not success or inliers is None:
            return None

        R_mat, _  = cv2.Rodrigues(rvec)
        t = tvec.reshape(3)

        return Tf.from_components(t, R.from_matrix(R_mat)), inliers
    
