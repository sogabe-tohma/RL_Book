import numpy as np
class ActorNet(object):
  def __init__(self, input_size, hidden_size1, hidden_size2,output_size):
    print("An actor network is created.")
    self.params = {}
    self.params['W1'] = self._uniform_init(input_size, hidden_size1)
    self.params['b1'] = np.zeros(hidden_size1)
    self.params['W2'] = self._uniform_init(hidden_size1, hidden_size2)
    self.params['b2'] = np.zeros(hidden_size2)
    self.params['W3'] = np.random.uniform(-3e-3, 3e-3, (hidden_size2, output_size))
    self.params['b3'] = np.zeros(output_size)
    self.params['W1_tgt'] = self._uniform_init(input_size, hidden_size1)
    self.params['b1_tgt'] = np.zeros(hidden_size1)
    self.params['W2_tgt'] = self._uniform_init(hidden_size1, hidden_size2)
    self.params['b2_tgt'] = np.zeros(hidden_size2)
    self.params['W3_tgt'] = np.random.uniform(-3e-3, 3e-3, (hidden_size2, output_size))
    self.params['b3_tgt'] = np.zeros(output_size)

    self.optm_cfg ={}
    self.optm_cfg['W1'] = None
    self.optm_cfg['b1'] = None
    self.optm_cfg['W2'] = None
    self.optm_cfg['b2'] = None
    self.optm_cfg['W3'] = None
    self.optm_cfg['b3'] = None

  def backpropagation(self, X, action_grads, action_bound, target=False):

    if not target:
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
    else:
        W1, b1 = self.params['W1_tgt'], self.params['b1_tgt']
        W2, b2 = self.params['W2_tgt'], self.params['b2_tgt']
        W3, b3 = self.params['W3_tgt'], self.params['b3_tgt']

    batch_size, _ = X.shape
    scores = None
    z1=np.dot(X,W1)+b1
    H1=np.maximum(0,z1) #Activation first layer
    z2=np.dot(H1,W2)+b2
    H2=np.maximum(0,z2) #Activation second layer
    scores=np.dot(H2, W3)+b3
    actions=np.tanh(scores)*action_bound
    grads = {}
    grad_output=action_bound*(1-np.tanh(scores)**2)*(-action_grads)
    out1=grad_output.dot(W3.T)
    out1[z2<=0]=0
    out2=out1.dot(W2.T)
    out2[z1<=0]=0
    grads['W3']=np.dot(H2.T, grad_output)/batch_size
    grads['W2']=np.dot(H1.T, out1)/batch_size
    grads['W1']=np.dot(X.T, out2)/batch_size
    grads['b3']=np.sum(grad_output, axis=0)/batch_size
    grads['b2']=np.sum(out1, axis=0)/batch_size
    grads['b1']=np.sum(out2, axis=0)/batch_size
    #print('this is grad form actor',grads['W1'])
    return actions, grads

  def weight_update(self, X, action_grads, action_bound):
    _, grads = self.backpropagation(X, action_grads, \
                                      action_bound)

    self.params['W3'] = self._adam(self.params['W3'], grads['W3'], config=self.optm_cfg['W3'])[0]
    self.params['W2'] = self._adam(self.params['W2'], grads['W2'], config=self.optm_cfg['W2'])[0]
    self.params['W1'] = self._adam(self.params['W1'], grads['W1'], config=self.optm_cfg['W1'])[0]
    self.params['b3'] = self._adam(self.params['b3'], grads['b3'], config=self.optm_cfg['b3'])[0]
    self.params['b2'] = self._adam(self.params['b2'], grads['b2'], config=self.optm_cfg['b2'])[0]
    self.params['b1'] = self._adam(self.params['b1'], grads['b1'], config=self.optm_cfg['b1'])[0]

    # Update the configuration parameters to be used in the next iteration
    self.optm_cfg['W3'] = self._adam(self.params['W3'], grads['W3'], config=self.optm_cfg['W3'])[1]
    self.optm_cfg['W2'] = self._adam(self.params['W2'], grads['W2'], config=self.optm_cfg['W2'])[1]
    self.optm_cfg['W1'] = self._adam(self.params['W1'], grads['W1'], config=self.optm_cfg['W1'])[1]
    self.optm_cfg['b3'] = self._adam(self.params['b3'], grads['b3'], config=self.optm_cfg['b3'])[1]
    self.optm_cfg['b2'] = self._adam(self.params['b2'], grads['b2'], config=self.optm_cfg['b2'])[1]
    self.optm_cfg['b1'] = self._adam(self.params['b1'], grads['b1'], config=self.optm_cfg['b1'])[1]




  def weight_update_target(self, tau):
    self.params['W3_tgt'] = tau*self.params['W3']+(1-tau)*self.params['W3_tgt']
    self.params['W2_tgt'] = tau*self.params['W2']+(1-tau)*self.params['W2_tgt']
    self.params['W1_tgt'] = tau*self.params['W1']+(1-tau)*self.params['W1_tgt']

    self.params['b3_tgt'] = tau*self.params['b3']+(1-tau)*self.params['b3_tgt']
    self.params['b2_tgt'] = tau*self.params['b2']+(1-tau)*self.params['b2_tgt']
    self.params['b1_tgt'] = tau*self.params['b1']+(1-tau)*self.params['b1_tgt']

  def predict(self, X, action_bound, target=False):
    y_pred = None

    if not target:

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
    else:

        W1, b1 = self.params['W1_tgt'], self.params['b1_tgt']
        W2, b2 = self.params['W2_tgt'], self.params['b2_tgt']
        W3, b3 = self.params['W3_tgt'], self.params['b3_tgt']

    H1=np.maximum(0,X.dot(W1)+b1)
    H2=np.maximum(0,H1.dot(W2)+b2)
    scores=H2.dot(W3)+b3
    y_pred=np.tanh(scores)*action_bound
    return y_pred

  def _adam(self, x, dx, config=None):
      if config is None: config = {}
      config.setdefault('learning_rate', 1e-4)
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
