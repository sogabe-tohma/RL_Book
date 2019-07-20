import numpy as np

class CriticNet(object):
  def __init__(self, input_size_S,input_size_A, hidden_size1, hidden_size2, \
               hidden_size3, output_size, weight=None):
    print("A critic network is created.")
    self.params = {}
    if weight == None:
        self.params['W1'] = self._uniform_init(input_size_S, hidden_size1)
        self.params['b1'] = np.zeros(hidden_size1)
        self.params['W2_S'] = self._uniform_init(hidden_size1, hidden_size2)
        self.params['b2_S'] = np.zeros(hidden_size2)
        self.params['W2_A'] = self._uniform_init(input_size_A, hidden_size2)
        self.params['b2_A'] = np.zeros(hidden_size2)
        self.params['W3'] = self._uniform_init(hidden_size2, hidden_size3)
        self.params['b3'] = np.zeros(hidden_size3)
        self.params['W4'] = np.random.uniform(-3e-3, 3e-3, (hidden_size3, output_size))
        self.params['b4'] = np.zeros(output_size)
    else:
        self.params['W1'] = weight['W1']
        self.params['b1'] = weight['b1']
        self.params['W2_S'] = weight['W2_S']
        self.params['b2_S'] = weight['b2_S']
        self.params['W2_A'] = weight['W2_A']
        self.params['b2_A'] = weight['b2_A']
        self.params['W3'] = weight['W3']
        self.params['b3'] = weight['b3']
        self.params['W4'] = weight['W4']
        self.params['b4'] = weight['b4']
    self.params['W1_tgt'] = self._uniform_init(input_size_S, hidden_size1)
    self.params['b1_tgt'] = np.zeros(hidden_size1)
    self.params['W2_S_tgt'] = self._uniform_init(hidden_size1, hidden_size2)
    self.params['b2_S_tgt'] = np.zeros(hidden_size2)
    self.params['W2_A_tgt'] = self._uniform_init(input_size_A, hidden_size2)
    self.params['b2_A_tgt'] = np.zeros(hidden_size2)
    self.params['W3_tgt'] = self._uniform_init(hidden_size2, hidden_size3)
    self.params['b3_tgt'] = np.zeros(hidden_size3)
    self.params['W4_tgt'] = np.random.uniform(-3e-3, 3e-3, (hidden_size3, output_size))
    self.params['b4_tgt'] = np.zeros(output_size)


    self.optm_cfg ={}
    self.optm_cfg['W1'] = None
    self.optm_cfg['b1'] = None
    self.optm_cfg['W2_S'] = None
    self.optm_cfg['b2_S'] = None
    self.optm_cfg['W2_A'] = None
    self.optm_cfg['b2_A'] = None
    self.optm_cfg['W3'] = None
    self.optm_cfg['b3'] = None
    self.optm_cfg['W4'] = None
    self.optm_cfg['b4'] = None



  def evaluate_gradient(self, X_S, X_A, Y_tgt, use_target=False):
    # Unpack variables from the params dictionary
    if not use_target:
        W1, b1 = self.params['W1'], self.params['b1']
        W2_S, b2_S = self.params['W2_S'], self.params['b2_S']
        W2_A, b2_A = self.params['W2_A'], self.params['b2_A']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']
    else:
        W1, b1 = self.params['W1_tgt'], self.params['b1_tgt']
        W2_S, b2_S = self.params['W2_S_tgt'], self.params['b2_S_tgt']
        W2_A, b2_A = self.params['W2_A_tgt'], self.params['b2_A_tgt']
        W3, b3 = self.params['W3_tgt'], self.params['b3_tgt']
        W4, b4 = self.params['W4_tgt'], self.params['b4_tgt']


    output= None
    z1=np.dot(X_S,W1)+b1
    H1=np.maximum(0,z1)
  
    z2_S=np.dot(H1,W2_S)+b2_S
    
    z2_A=np.dot(X_A,W2_A)+b2_A
    H2=z2_S + z2_A # No relu here
    z3=np.dot(H2,W3)+b3
    H3=np.maximum(0,z3) # Relu here
    output=np.dot(H3, W4)+b4
    Q_values=output # q_values is the scores in this case, due to the linear
    batch_size=np.shape(X_S)[0]
     # loss
    loss =np.sum((Q_values-Y_tgt)**2, axis=0)/(1.0*batch_size)
    loss = loss[0]
    # error
    error=(Q_values-Y_tgt)
    grads = {}
    grad_output=error*2/batch_size
    out1=grad_output.dot(W4.T)
    out1[z3<=0]=0
    out2=out1.dot(W3.T)
    out3=out2.dot(W2_S.T)
    out3[z1<=0]=0
    # Calculate gradient using back propagation
    grads['W4']=np.dot(H3.T, grad_output)/(1.0*batch_size)
    grads['W3']=np.dot(H2.T, out1)/(1.0*batch_size)
    grads['W2_S']=np.dot(H1.T, out2)/(1.0*batch_size)
    grads['W2_A']=np.dot(X_A.T, out2)/(1.0*batch_size)
    grads['W1']=np.dot(X_S.T, out3)/(1.0*batch_size)
    grads['b4']=np.sum(grad_output, axis=0)/(1.0*batch_size)
    grads['b3']=np.sum(out1, axis=0)/(1.0*batch_size)
    grads['b2_S']=np.sum(out2, axis=0)/(1.0*batch_size)
    grads['b2_A']=np.sum(out2, axis=0)/(1.0*batch_size)
    grads['b1']=np.sum(out3, axis=0)/(1.0*batch_size)


    return grads,loss #, Q_values


  def evaluate_action_gradient(self, X_S, X_A, use_target=False):
    if not use_target:
        W1, b1 = self.params['W1'], self.params['b1']
        W2_S, b2_S = self.params['W2_S'], self.params['b2_S']
        W2_A, b2_A = self.params['W2_A'], self.params['b2_A']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']
    else:
        W1, b1 = self.params['W1_tgt'], self.params['b1_tgt']
        W2_S, b2_S = self.params['W2_S_tgt'], self.params['b2_S_tgt']
        W2_A, b2_A = self.params['W2_A_tgt'], self.params['b2_A_tgt']
        W3, b3 = self.params['W3_tgt'], self.params['b3_tgt']
        W4, b4 = self.params['W4_tgt'], self.params['b4_tgt']
    output= None
    z1=np.dot(X_S,W1)+b1
    H1=np.maximum(0,z1)
    z2_S=np.dot(H1,W2_S)+b2_S
    z2_A=np.dot(X_A,W2_A)+b2_A
    H2=z2_S + z2_A # No relu here
    z3=np.dot(H2,W3)+b3
    H3=np.maximum(0,z3)
    output=np.dot(H3, W4)+b4
    grad_output=np.ones_like(output)
    out1=grad_output.dot(W4.T)
    out1[z3<=0]=0
    out2=out1.dot(W3.T)
    grads_action = np.dot(out2,W2_A.T)
    return grads_action



  def weight_update(self, X_S, X_A, Y_tgt):
    grads, loss = self.evaluate_gradient(X_S, X_A, Y_tgt, use_target=False)
    self.params['W4'] = self._adam(self.params['W4'], grads['W4'], config=self.optm_cfg['W4'])[0]
    self.params['W3'] = self._adam(self.params['W3'], grads['W3'], config=self.optm_cfg['W3'])[0]
    self.params['W2_S'] = self._adam(self.params['W2_S'], grads['W2_S'], config=self.optm_cfg['W2_S'])[0]
    self.params['W2_A'] = self._adam(self.params['W2_A'], grads['W2_A'], config=self.optm_cfg['W2_A'])[0]
    self.params['W1'] = self._adam(self.params['W1'], grads['W1'], config=self.optm_cfg['W1'])[0]
    self.params['b4'] = self._adam(self.params['b4'], grads['b4'], config=self.optm_cfg['b4'])[0]
    self.params['b3'] = self._adam(self.params['b3'], grads['b3'], config=self.optm_cfg['b3'])[0]
    self.params['b2_S'] = self._adam(self.params['b2_S'], grads['b2_S'], config=self.optm_cfg['b2_S'])[0]
    self.params['b2_A'] = self._adam(self.params['b2_A'], grads['b2_A'], config=self.optm_cfg['b2_A'])[0]
    self.params['b1'] = self._adam(self.params['b1'], grads['b1'], config=self.optm_cfg['b1'])[0]
    # Update the configuration parameters to be used in the next iteration
    self.optm_cfg['W4'] = self._adam(self.params['W4'], grads['W4'], config=self.optm_cfg['W4'])[1]
    self.optm_cfg['W3'] = self._adam(self.params['W3'], grads['W3'], config=self.optm_cfg['W3'])[1]
    self.optm_cfg['W2_S'] = self._adam(self.params['W2_S'], grads['W2_S'], config=self.optm_cfg['W2_S'])[1]
    self.optm_cfg['W2_A'] = self._adam(self.params['W2_A'], grads['W2_A'], config=self.optm_cfg['W2_A'])[1]
    self.optm_cfg['W1'] = self._adam(self.params['W1'], grads['W1'], config=self.optm_cfg['W1'])[1]
    self.optm_cfg['b4'] = self._adam(self.params['b4'], grads['b4'], config=self.optm_cfg['b4'])[1]
    self.optm_cfg['b3'] = self._adam(self.params['b3'], grads['b3'], config=self.optm_cfg['b3'])[1]
    self.optm_cfg['b2_S'] = self._adam(self.params['b2_S'], grads['b2_S'], config=self.optm_cfg['b2_S'])[1]
    self.optm_cfg['b2_A'] = self._adam(self.params['b2_A'], grads['b2_A'], config=self.optm_cfg['b2_A'])[1]
    self.optm_cfg['b1'] = self._adam(self.params['b1'], grads['b1'], config=self.optm_cfg['b1'])[1]

    return loss
  def weight_update_target(self, tau):
    self.params['W4_tgt'] = tau*self.params['W4']+(1-tau)*self.params['W4_tgt']
    self.params['W3_tgt'] = tau*self.params['W3']+(1-tau)*self.params['W3_tgt']
    self.params['W2_S_tgt'] = tau*self.params['W2_S']+(1-tau)*self.params['W2_S_tgt']
    self.params['W2_A_tgt'] = tau*self.params['W2_A']+(1-tau)*self.params['W2_A_tgt']
    self.params['W1_tgt'] = tau*self.params['W1']+(1-tau)*self.params['W1_tgt']

    self.params['b4_tgt'] = tau*self.params['b4']+(1-tau)*self.params['b4_tgt']
    self.params['b3_tgt'] = tau*self.params['b3']+(1-tau)*self.params['b3_tgt']
    self.params['b2_S_tgt'] = tau*self.params['b2_S']+(1-tau)*self.params['b2_S_tgt']
    self.params['b2_A_tgt'] = tau*self.params['b2_A']+(1-tau)*self.params['b2_A_tgt']
    self.params['b1_tgt'] = tau*self.params['b1']+(1-tau)*self.params['b1_tgt']

  def predict(self, X_S, X_A, target=False):
    y_pred = None

    if not target:
        W1, b1 = self.params['W1'], self.params['b1']
        W2_S, b2_S = self.params['W2_S'], self.params['b2_S']
        W2_A, b2_A = self.params['W2_A'], self.params['b2_A']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']

    else:
        W1, b1 = self.params['W1_tgt'], self.params['b1_tgt']
        W2_S, b2_S = self.params['W2_S_tgt'], self.params['b2_S_tgt']
        W2_A, b2_A = self.params['W2_A_tgt'], self.params['b2_A_tgt']
        W3, b3 = self.params['W3_tgt'], self.params['b3_tgt']
        W4, b4 = self.params['W4_tgt'], self.params['b4_tgt']

    H1 = np.maximum(0,X_S.dot(W1)+b1)

    H2 = np.dot(H1,W2_S)+b2_S + np.dot(X_A, W2_A)+b2_A

    H3 = np.maximum(0,H2.dot(W3)+b3)

    score=np.dot(H3, W4)+b4
    y_pred=score
    return y_pred

  def _adam(self, x, dx, config=None):
      if config is None: config = {}
      config.setdefault('learning_rate', 1e-3)
      config.setdefault('beta1', 0.9)
      config.setdefault('beta2', 0.999)
      config.setdefault('epsilon', 1e-8)
      config.setdefault('m', np.zeros_like(x))
      config.setdefault('v', np.zeros_like(x))
      config.setdefault('t', 0)

      next_x = None

      #Adam update formula,                                                 #
      config['t'] += 1
      config['m'] = config['beta1']*config['m'] + (1-config['beta1'])*dx
      config['v'] = config['beta2']*config['v'] + (1-config['beta2'])*(dx**2)
      mb = config['m'] / (1 - config['beta1']**config['t'])
      vb = config['v'] / (1 - config['beta2']**config['t'])

      next_x = x - config['learning_rate'] * mb / (np.sqrt(vb) + config['epsilon'])
      return next_x, config

  def _uniform_init(self, input_size, output_size):
      u = np.sqrt(6./(input_size+output_size))
      return np.random.uniform(-u, u, (input_size, output_size))
