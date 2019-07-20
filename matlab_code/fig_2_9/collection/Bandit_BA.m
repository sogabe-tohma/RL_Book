classdef Bandit_BA <handle
    %   Detailed explanation goes here
    properties
        N_arm,mean_rate,arm_values,k,est_values,lambda_,sum_x,tau,sample
        
    end    
    methods
        function self=init(self,N_arm)
            self.N_arm=N_arm;
            self.lambda_=ones(1,N_arm);
            self.mean_rate=0.0;
            self.arm_values=normrnd(0,1,N_arm);
            self.k=zeros(1,N_arm);
            self.est_values=ones(1,N_arm)* self.mean_rate;
            self.sum_x=zeros(1,N_arm);
            self.tau=ones(1,N_arm);
            self.sample=zeros(1,N_arm);
        end
        function reward = get_reward(self,action)
            noise= normrnd(0,1);
            reward=(self.arm_values(action))+noise;
         
        end
        function idx=choose_eps_greedy(self)
            self.sample= randn(1)./sqrt(self.lambda_)+self.est_values;        
            [~,idx]=max(self.sample);
           
        end    
        function val=update_est(self,action,x)
            self.lambda_(action)=self.lambda_(action)+1;
            self.sum_x(action)=self.sum_x(action)+x;
            self.est_values(action)=self.tau(action)*self.sum_x(action)/self.lambda_(action);
            val=self.est_values(action);          
        end
    end
end

