classdef cr <handle
    properties(SetAccess=public)
        c_w={unform( 2,30),unform(30,60),unform(1,60),unform(60,60),random('Uniform',-3e-3, 3e-3, [60,1])};
        c_b={zeros(1,30),zeros(1,60),zeros(1,60),zeros(1,60),zeros(1)};
        ct_w={unform( 2,30),unform(30,60),unform(1,60),unform(60,60),random('Uniform',-3e-3, 3e-3, [60,1])};
        ct_b={zeros(1,30),zeros(1,60),zeros(1,60),zeros(1,60),zeros(1)};
        opt_W={{zeros(1,3)},{zeros(1,3)},{zeros(1,3)},{zeros(1,3)},{zeros(1,3)}};
        opt_B={{zeros(1,3)},{zeros(1,3)},{zeros(1,3)},{zeros(1,3)},{zeros(1,3)}};
        grad= {};
        
    end   
    methods    
        function [los]=evaluate_gradient(self, X_s,X_a,y_tgt,targ)
           
            if targ==1
                cw1= self.c_w{1};
                cb1= self.c_b{1};
                cw2_s= self.c_w{2};
                cb2_s= self.c_b{2};
                cw2_a= self.c_w{3};
                cb2_a= self.c_b{3};
                cw3= self.c_w{4};
                cb3= self.c_b{4};
                cw4= self.c_w{5};
                cb4= self.c_b{5};
            else
                cw1= self.ct_w{1};
                cb1= self.ct_b{1};
                cw2_s= self.ct_w{2};
                cb2_s= self.ct_b{2};
                cw2_a= self.ct_w{3};
                cb2_a= self.ct_b{3};
                cw3= self.ct_w{4};
                cb3= self.ct_b{4};
                cw4= self.ct_w{5};
                cb4= self.ct_b{5};
            end
            cz1=(X_s*cw1)+cb1;
            cH1=max(0,cz1);
            z2_s=(cH1*cw2_s)+cb2_s;
            z2_a=(X_a*cw2_a)+cb2_a;
            cH2=z2_s+z2_a;
            cz3=(cH2*cw3)+cb3; 
            cH3=max(0,cz3);
            cscores=(cH3*cw4)+cb4;
            Q_value=cscores;
            cbatch_size=length(X_s(1));
            los=sum(Q_value-y_tgt).^2./(1*cbatch_size);
            error= Q_value-y_tgt;
            grad_output= (error.*2)./cbatch_size;
            cout1=(grad_output*cw4');
            cout1(cz3<=0)=0;
            cout2=(cout1*cw3');
            cout3=(cout2*cw2_s');
            self.grad.w5=(cH3'* grad_output)/(1.0*cbatch_size);
            self.grad.w4=(cH2'* cout1)/(1.0*cbatch_size);
            self.grad.w2=(cH1' *cout2 )/(1.0*cbatch_size);
            self.grad.w3=(X_a'*cout2)/(1.0*cbatch_size);
            self.grad.w1=(X_s'*cout3)/(1.0*cbatch_size);
            self.grad.b5=sum(grad_output, 1)/(1.0*cbatch_size);
            self.grad.b4=sum(cout1, 1)/(1.0*cbatch_size);
            self.grad.b2=sum(cout2, 1)/(1.0*cbatch_size);
            self.grad.b3=sum(cout2, 1)/(1.0*cbatch_size);
            self.grad.b1=sum(cout3, 1)/(1.0*cbatch_size);

        end
        function los=train(self,X_s,X_a,y_tgt)
            [los] = self.evaluate_gradient(X_s,X_a,y_tgt,1);
            [self.c_w{5},self.opt_W{5}] = adam(self.c_w{5}, self.grad.w5, self.opt_W{5},length(self.opt_W{5}));
            [self.c_w{4},self.opt_W{4}] = adam(self.c_w{4}, self.grad.w4, self.opt_W{4},length(self.opt_W{4}));
            [self.c_w{3},self.opt_W{3}] = adam(self.c_w{3}, self.grad.w3, self.opt_W{3},length(self.opt_W{3}));
            [self.c_w{2},self.opt_W{2}] = adam(self.c_w{2}, self.grad.w2, self.opt_W{2},length(self.opt_W{2}));
            [self.c_w{1},self.opt_W{1}] = adam(self.c_w{1}, self.grad.w1, self.opt_W{1},length(self.opt_W{1}));
            [self.c_b{5},self.opt_B{5}] = adam(self.c_b{5}, self.grad.b5, self.opt_B{5},length(self.opt_B{5}));
            [self.c_b{4},self.opt_B{4}] = adam(self.c_b{4}, self.grad.b4, self.opt_B{4},length(self.opt_B{4}));
            [self.c_b{3},self.opt_B{3}] = adam(self.c_b{3}, self.grad.b3, self.opt_B{3},length(self.opt_B{3}));
            [self.c_b{2},self.opt_B{2}] = adam(self.c_b{2}, self.grad.b2, self.opt_B{2},length(self.opt_B{2}));
            [self.c_b{1},self.opt_B{1}] = adam(self.c_b{1}, self.grad.b1, self.opt_B{1},length(self.opt_B{1}));
        end
        function train_target(self, tau)
    
            self.ct_w{5} = tau*self.c_w{5}+(1-tau)*self.ct_w{5};
            self.ct_w{4} = tau*self.c_w{4}+(1-tau)*self.ct_w{4};
            self.ct_w{3} = tau*self.c_w{3}+(1-tau)*self.ct_w{3};
            self.ct_w{2} = tau*self.c_w{2}+(1-tau)*self.ct_w{2};
            self.ct_w{1} = tau*self.c_w{1}+(1-tau)*self.ct_w{1};
            
            self.ct_b{5} = tau*self.c_b{5}+(1-tau)*self.ct_b{5};
            self.ct_b{4} = tau*self.c_b{4}+(1-tau)*self.ct_b{4};
            self.ct_b{3} = tau*self.c_b{3}+(1-tau)*self.ct_b{3};
            self.ct_b{2} = tau*self.c_b{2}+(1-tau)*self.ct_b{2};
            self.ct_b{1} = tau*self.c_b{1}+(1-tau)*self.ct_b{1};
        end
        function y_pre =predict(self,X_s,X_a,GeT)
            y_pre = 0;
            if GeT==1
                cW1 = self.c_w{1};
                cb1 = self.c_b{1};
                W2_s = self.c_w{2};
                b2_s = self.c_b{2};
                W2_a = self.c_w{3};
                b2_a = self.c_b{3};
                cW3 = self.c_w{4};
                cb3 = self.c_b{4};
                cW4 = self.c_w{5};
                cb4 = self.c_b{5};
            else
                cW1 = self.ct_w{1};
                cb1 = self.ct_b{1};
                W2_s = self.ct_w{2};
                b2_s = self.ct_b{2};
                W2_a = self.ct_w{3};
                b2_a = self.ct_b{3};
                cW3 = self.ct_w{4};
                cb3 = self.ct_b{4};
                cW4 = self.ct_w{5};
                cb4 = self.ct_b{5};
            end
            zc1=(X_s*cW1)+cb1; 
            Hc1=max(0,zc1);
            cz2_s=(Hc1*W2_s)+b2_s;
            cz2_a=(X_a*W2_a)+b2_a;
            Hc2=cz2_s+cz2_a;
            zc3=(Hc2*cW3)+cb3; 
            Hc3=max(0,zc3);
            scoe=(Hc3*cW4)+cb4; 
            y_pre=scoe;
            
        end
        function [grand_action]=evaluate_action_gradient(self,X_s,X_a,tar)
            if tar==1
                Wc1 = self.c_w{1};
                bc1 = self.c_b{1};
                Wc2_s = self.c_w{2};
                bc2_s = self.c_b{2};
                Wc2_a = self.c_w{3};
                bc2_a = self.c_b{3};
                Wc3 = self.c_w{4};
                bc3 = self.c_b{4};
                Wc4 = self.c_w{5};
                bc4 = self.c_b{5};
            else
                Wc1 = self.ct_w{1};
                bc1 = self.ct_b{1};
                Wc2_s = self.ct_w{2};
                bc2_s = self.ct_b{2};
                Wc2_a = self.ct_w{3};
                bc2_a = self.ct_b{3};
                Wc3 = self.ct_w{4};
                bc3 = self.ct_b{4};
                Wc4 = self.ct_w{5};
                bc4 = self.ct_b{5};
            end
            coutput=0;
            c_z1=(X_s*Wc1)+bc1; 
            c_H1=max(0,c_z1);
            c_z2_s=(c_H1*Wc2_s)+bc2_s;
            c_z2_a=(X_a*Wc2_a)+bc2_a;
            c_H2=c_z2_s+c_z2_a;
            c_z3=(c_H2*Wc3)+bc3; 
            c_H3=max(0,c_z3);
            coutput=(c_H3*Wc4)+bc4;
            grand_out=ones(size(coutput),'like',coutput);
            c_out1=(grand_out*Wc4');
            c_out1(c_z3<=0)=0;
            c_out2=(c_out1*Wc3');
            grand_action=(c_out2*Wc2_a');        
            
        end
        function y= rel(X) 
            y=tanh(X);
        end
        function yy=sigmoid(XX)
            yy= (1/(1+exp(XX)));
        end
       
    end
    
end

 function y= tan(X) 
            y=tanh(X);
 end 
function[next,data] = adam(x, dx,con,make)            
            if make== 1
                config.learning_rate= 1e-3;
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
            
            next_x =zeros(size(x)) ;

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
