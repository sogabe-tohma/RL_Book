classdef Bandit_initial_value <handle
    %   Detailed explanation goes here
    properties
        t=normrnd(0,1,10);
        mean=0;
        N=0;
        seed=RandStream.setGlobalStream(RandStream('mt19937ar','Seed',35678));
    end
    
    methods
        
        function n_pull = pull(self,X)
            %fprintf('%d  %f \n', X,self.t(X));
            n_pull = self.t(X)+normrnd(0,1);
        end
        
        function update(self,x)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            self.N = self.N+1;
            self.mean=(1.0-1.0/self.N)*self.mean+(1.0/self.N)*x;
            
        end
    end
end

