classdef actor <handle
    properties(SetAccess=public)
        a_w={unform( 2,300),unform(300,600),random('Uniform',-3e-3, 3e-3, [600,1])};
        a_b={zeros(1,300),zeros(1,600),zeros(1)};
        at_w={unform( 2,300),unform(300,600),random('Uniform',-3e-3, 3e-3, [600,1])};
        at_b={zeros(1,300),zeros(1,600),zeros(1)};
        opt_w={{zeros(1,3)},{zeros(1,3)},{zeros(1,3)}};
        opt_b={{zeros(1,3)},{zeros(1,3)},{zeros(1,3)}};
        grads = {};
        
    end   
    methods    
        function [actions]=evaluate_gradient(self, X, action_grads, target)
            batch_size=2;
            if target==1
                w1= self.a_w{1};
                b1= self.a_b{1};
                w2= self.a_w{2};
                b2= self.a_b{2};
                w3= self.a_w{3};
                b3= self.a_b{3};
            else
                w1= self.at_w{1};
                b1= self.at_b{1};
                w2= self.at_w{2};
                b2= self.at_b{2};
                w3= self.at_w{3};
                b3= self.at_b{3};
            end
            scores = 0;
            
            z1=(X*w1)+b1; 
            H1=max(0,z1);
            z2=(H1*w2)+b2;
            H2=max(0,z2);
            scores=(H2*w3)+b3;
            actions=tanh(scores)*2; 
           
            grad_output= 2*(1-tanh(scores).^2).*(-action_grads);
            out1=(grad_output*w3');
            out1(z2<=0)=0;
            out2=(out1*w2');
            out2(z1<=0)=0;
            self.grads.w3=(H2'* grad_output)/batch_size;
            self.grads.w2=(H1' *out1 )/batch_size;
            self.grads.w1=(X'*out2)/batch_size;
            self.grads.b3=sum(grad_output, 1)/batch_size;
            self.grads.b2=sum(out1, 1)/batch_size;
            self.grads.b1=sum(out2, 1)/batch_size;

        end
        function train(self, X, action_grads,use)
            [~] = self.evaluate_gradient(X, action_grads,use);
            
            [self.a_w{3},self.opt_w{3}] = adam(self.a_w{3}, self.grads.w3, self.opt_w{3},length(self.opt_w{3}));
            [self.a_w{2},self.opt_w{2}] = adam(self.a_w{2}, self.grads.w2, self.opt_w{2},length(self.opt_w{2}));
            [self.a_w{1},self.opt_w{1}] = adam(self.a_w{1}, self.grads.w1, self.opt_w{1},length(self.opt_w{1}));
            [self.a_b{3},self.opt_b{3}] = adam(self.a_b{3}, self.grads.b3, self.opt_b{3},length(self.opt_b{3}));
            [self.a_b{2},self.opt_b{2}] = adam(self.a_b{2}, self.grads.b2, self.opt_b{2},length(self.opt_b{2}));
            [self.a_b{1},self.opt_b{1}] = adam(self.a_b{1}, self.grads.b1, self.opt_b{1},length(self.opt_b{1}));
        end
        function train_target(self, tau)
    
            self.at_w{3} = tau*self.a_w{3}+(1-tau)*self.at_w{3};
            self.at_w{2} = tau*self.a_w{2}+(1-tau)*self.at_w{2};
            self.at_w{1} = tau*self.a_w{1}+(1-tau)*self.at_w{1};
            
            self.at_b{3} = tau*self.a_b{3}+(1-tau)*self.at_b{3};
            self.at_b{2} = tau*self.a_b{2}+(1-tau)*self.at_b{2};
            self.at_b{1} = tau*self.a_b{1}+(1-tau)*self.at_b{1};
        end
        function y_pre =predict(self, X, GeT)
            y_pre = 0;
            if GeT==1
                W1 = self.a_w{1};
                b_1 = self.a_b{1};
                W2 = self.a_w{2};
                b_2 = self.a_b{2};
                W3 = self.a_w{3};
                b_3 = self.a_b{3};
            else
                W1 = self.at_w{1};
                b_1 = self.at_b{1};
                W2 = self.at_w{2};
                b_2 = self.at_b{2};
                W3 = self.at_w{3};
                b_3 = self.at_b{3};
            end
          
            z_1=(X*W1)+b_1; 
            H_1=max(0,z_1);
            z_2=(H_1*W2)+b_2;
            H_2=max(0,z_2);
            scor=(H_2*W3)+b_3;
            y_pre=tanh(scor)*2;
           
            
        end
       
    end
    
end
function[next,data] = adam(x, dx,con,make)            
            if make== 1
                config.learning_rate= 1e-4;
                config.beta1= 0.9;
                config.beta2= 0.999;
                config.epsilon= 1e-8;
                config.m= zeros(size(x),'like',x);
                config.v= zeros(size(x),'like',x);
                config.t= 0;
            else
                config.learning_rate= 1e-4;
                config.beta1= 0.9;
                config.beta2= 0.999;
                config.epsilon= 1e-8;
                config.m= con{2};
                config.v= con{3};
                config.t= con{1};
            end
            config.t= config.t+1;
            config.m= config.beta1*config.m + (1-config.beta1)*dx;
            config.v = config.beta2*config.v + (1-config.beta2)*(dx.^2);
            mb = config.m/ (1 - config.beta1.^config.t);
            vb = config.v / (1 - config.beta2.^config.t);
            next= x - config.learning_rate .* mb ./ (sqrt(vb) + config.epsilon);
            data={config.t,config.m,config.v};
end
function [weight]= unform( input_size, output_size)
     u =sqrt(6./(input_size+output_size));
     weight= random('Uniform',-u, u, [input_size,output_size]);
end