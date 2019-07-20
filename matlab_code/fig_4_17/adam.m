function[next,config] =adam(x, dx, config)
if config == 1
    config = {};
end
config.learning_rate= 1e-4;
config.beta1= 0.9;
config.beta2= 0.999;
config.epsilon= 1e-8;
config.m= zeros(size(x),'like',x);
config.v= zeros(size(x),'like',x);
config.t= 0;
next_x =zeros(size(x)) ;
                                 
config.t= config.t+1;
config.m= config.beta1*config.m + (1-config.beta1)*dx;
config.v = config.beta2*config.v + (1-config.beta2)*(dx.^2);
mb = config.m/ (1 - config.beta1.^config.t);
vb = config.v / (1 - config.beta2.^config.t);
next= x - config.learning_rate .* mb ./ (sqrt(vb) + config.epsilon);
end