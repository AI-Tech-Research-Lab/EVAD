import numpy as np
import tenseal as ts

def preprocess_a_sample(df, windows):
    final_sample = []
    
    for column in df.columns:
        signal = df.loc[:, column]
        
        # signal = signal - signal.mean()
        signal_fft = np.abs(np.fft.rfft(signal))**2
        # transformer = RobustScaler()
        # signal_fft = transformer.fit_transform(signal_fft.reshape(-1, 1)).reshape(1, -1)[0]
        # signal_fft = np.abs(np.fft.rfft(signal))
        len_windows = int(len(signal_fft) / windows) - 1
        
        for i in range(windows):
            if i == windows-1:
                final_sample.append(np.mean(signal_fft[i*len_windows:]))
            else:
                final_sample.append(np.mean(signal_fft[i*len_windows:(i+1)*len_windows]))
                
    return np.array(final_sample)


def preprocess_a_sample_encrypted(sample, context, windows, scaler):
    def create_fourier_weights(signal_length):  
        "Create weights, as described above. https://sidsite.com/posts/fourier-nets/"

        dim_output = (signal_length/2) + 1 if signal_length % 2 == 0 else (signal_length+1) / 2

        k_vals, n_vals = np.mgrid[0:signal_length, 0:dim_output]
        theta_vals = 2 * np.pi * k_vals * n_vals / signal_length
        return np.hstack([np.cos(theta_vals), -np.sin(theta_vals)])
        

    def create_rotation_matrix(signal_length):
        d = (signal_length/2) + 1 if signal_length % 2 == 0 else (signal_length+1) / 2
        d = int(d)
        m = np.vstack([np.diag(np.full(d,1)), np.diag(np.full(d,1))])

        return m


    def create_windows_matrix(n_windows, windows_length, fft_length):
        for i in range(0, n_windows):

            if i != n_windows-1:
                _windows_length = windows_length
            else:
                _windows_length = fft_length - (n_windows-1)*windows_length

            sub = np.zeros([_windows_length, n_windows])
            sub[:, i] = 1/_windows_length
            if i == 0:
                m = sub
            else:
                m = np.vstack([m, sub])
        return m


    def create_stack_matrices(axes, windows_length):
        zeros = np.zeros((windows_length, windows_length))
        diag = np.diag(np.full(windows_length, 1))

        if axes == 2:
            return (np.hstack([diag, zeros]), np.hstack([zeros, diag]))
        
        if axes == 3:
            return (np.hstack([diag, zeros, zeros]), np.hstack([zeros, diag, zeros]), np.hstack([zeros, zeros, diag]))


    df = sample
    n_axes = len(df.columns)

    signal_length = len(df)
    fft_length = int((signal_length / 2)) + 1
    len_windows = int(fft_length / windows) - 1

    encrypted_vectors = []
    for i in range(0, n_axes):
        encrypted_vectors.append(ts.ckks_vector(context, df.iloc[:, i]))

    W_fourier = create_fourier_weights(signal_length)
    W_rotation = create_rotation_matrix(signal_length)
    W_windows = create_windows_matrix(windows, len_windows, fft_length)
    
    W = W_rotation @ W_windows
    
    tuple_of_stack_matrices = create_stack_matrices(n_axes, windows)

    stack_matrices = []
    for i in range(0, n_axes):
        stack_matrices.append(W @ tuple_of_stack_matrices[i])
        
    for i in range(0, n_axes):
        encrypted_vectors[i] = ((encrypted_vectors[i] @ W_fourier) ** 2) @ stack_matrices[i]

    final_encrypted_sample = sum(encrypted_vectors)
    
    if scaler:
        size = final_encrypted_sample.size()
        final_encrypted_sample *= [scaler.scale_[0] for _ in range(0, size)]
        final_encrypted_sample += [scaler.min_[0] for _ in range(0, size)]
    
    return final_encrypted_sample
    

def he_svm(encrypted_sample, svm, windows):
    """
    """
    # with CodeTimer('Computing the SVM output'):
    SV = svm.support_vectors_
    gamma = svm.gamma_value
    degree = svm.degree

    DC = svm.dual_coef_
    intercept = svm.intercept_    
    enc_result = np.array([encrypted_sample.dot(SV[x] * gamma) for x in range(0, len(SV))])
    # enc_result = enc_result * gamma
    enc_result = enc_result ** degree
    enc_result = DC.dot(enc_result)
    enc_result = enc_result + intercept
    
    return enc_result
