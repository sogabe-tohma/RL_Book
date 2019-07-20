rng default;
clear all;
cart=cart_pole;
anet=actor_net(); % initialize actor net()
cnet= critic_net();% initialize critic net()
GAMMA   = 0.99;      % Discount factor for reward calculation 
ACTIONS = 2;      
MAX_STEPS = 500;
failures=0;
success=0;
copts.use_gpu=0;
aopts.use_gpu=0;
copts.parameters.mom =0.2;
copts.parameters.lr =1e-4;
aopts.parameters.mom =0.2;
aopts.parameters.lr =1e-4;
for increment=1:1000   
    % initialize the input and dzdy for gradient and state is initialize
    % fron cartpole calss which is also random 
    Input=zeros(4,MAX_STEPS);
    next_input=zeros(4,MAX_STEPS);
    copts.dzdy =zeros(1,MAX_STEPS);
    aopts.dzdy =zeros(2,MAX_STEPS);
    state=cart.re_set();
    samples=1;
    valid=0;
    while ~(valid)
        Input(:,samples)=state;
        res_a(1).x=state';
        res_c(1).x=state';
        [ anet,res_a,aopts] = net_ff(anet,res_a,aopts);
        Q_new=soft_max(res_a(end).x); 
        x1= round(Q_new(1),1);
        x2= round(Q_new(2),1);
        a_new=randsrc(1,1,[1,2;x1,x2]);
        %[~,a_new]=max(Q_new);
        [statet,r,valid]=cart.forward(a_new,increment) ; 
        % here valid mean game's continuity if game is over valid is 1     
        next_input(:,samples)=statet;
        res_ct(1).x=statet';
        [cnet,res_ct,copts] = net_ff(cnet,res_ct,copts);
        vt=res_ct(end).x;
        [cnet,res_c,copts] = net_ff(cnet,res_c,copts);
        v=res_c(end).x;
        der=r+ GAMMA * vt-v;
        copts.dzdy(1,samples)=der;
        aopts.dzdy(a_new,samples)=-1*der* Q_new(a_new);
        [ cnet,res_c,copts ] = net_bp( cnet,res_c,copts );    
        [ cnet,res_c,copts ] = adam( cnet,res_c,copts );
        [ anet,res_a,aopts ] = net_bp(anet,res_a,aopts );    
        [ anet,res_a,aopts ] = adam(anet,res_a,aopts );
        state=statet;
        samples=samples+1;
    end 
    % to display the running times and total reward for the episode
    disp(['Trial is ' int2str(increment) ' reward is '  num2str(samples-1)]);

end
%% it is mainly for the network making purpose
function net = actor_net()
rng('default');
f=1/1 ;
net.layers = {} ;
% 2-layer net
net.layers{end+1} = struct('type', 'mlp', ...
                           'weights', {{f*randn(40,4, 'single'), zeros(40,1,'single')}}) ;
net.layers{end+1} = struct('type', 'tanh') ;

net.layers{end+1} = struct('type', 'mlp', ...
                           'weights', {{f*randn(2,40, 'single'), zeros(2,1,'single')}}) ;
end
%% it is mainly for the network making purpose
function net = critic_net()
rng('default');
f=1/10 ;
net.layers = {} ;
% 2-layer net
net.layers{end+1} = struct('type', 'mlp', ...
                           'weights', {{f*randn(16,4, 'single'), zeros(16,1,'single')}}) ;
net.layers{end+1} = struct('type', 'tanh') ;

net.layers{end+1} = struct('type', 'mlp', ...
                           'weights', {{f*randn(1,16, 'single'), zeros(1,1,'single')}}) ;
end


%% it is mainly for forward network calculation purpose
function [ net,res,opts ] = net_ff( net,res,opts )
    if ~isfield(opts,'datatype')
        opts.datatype='single';
    end
    res(1).x=cast(res(1).x,opts.datatype);
    for layer=1:numel(net.layers)
        opts.current_layer=layer;        
        switch net.layers{layer}.type
            case {'mlp','linear'} 
                [res(layer+1).x,~,~,opts] = linear_layer( res(layer).x,net.layers{layer}.weights{1},net.layers{layer}.weights{2},[], opts );
            case {'normalize', 'lrn'}
                [res(layer+1).x,opts] = lrn(res(layer).x, net.layers{layer}.param(1),net.layers{layer}.param(2),net.layers{layer}.param(3),net.layers{layer}.param(4),[],opts) ;            
            case 'relu'
                res(layer+1).x = relu(res(layer).x,[] );
            case 'softmax'        
                res(layer+1).x = softmax(res(layer).x,[]) ;
            case 'tanh'
                res(layer+1).x = tanh_ln(res(layer).x,[] );
            otherwise 
                error('net_ff error')                
        end
    end
end
function y = relu(x,dzdy)
  if nargin <= 1 || isempty(dzdy)
    y = max(x, single(0)) ;
  else
    y = dzdy .* (x > single(0)) ;
  end
end
function y = tanh_ln(x,dzdy)

    if nargin <= 1 || isempty(dzdy)
        y = tanh(x);
    else
        y = dzdy.*(4./(exp(x)+exp(-x)).^2);
    end

end
%% it is mainly for forward  and back propagation calculation purpose
function [ net,res,opts ] = net_bp( net,res,opts )
    if ~isfield(opts,'datatype')
        opts.datatype='single';
    end
    opts.dzdy=cast(opts.dzdy,opts.datatype);
    res(numel(net.layers)+1).dzdx = opts.dzdy ;
    if opts.use_gpu
         res(numel(net.layers)+1).dzdx=gpuArray( res(numel(net.layers)+1).dzdx);
    end       
    for layer=numel(net.layers):-1:1
       opts.current_layer=layer;
        switch net.layers{layer}.type
            case {'mlp','linear'}                             
                [res(layer).dzdx, res(layer).dzdw,res(layer).dzdb] = linear_layer( res(layer).x,net.layers{layer}.weights{1},net.layers{layer}.weights{2},res(layer+1).dzdx,opts );
            case 'relu'
                res(layer).dzdx = relu(res(layer).x, res(layer+1).dzdx) ;           
            case 'softmax'        
                res(layer).dzdx = softmax(res(layer).x,res(layer+1).dzdx) ;
            case 'tanh'
                res(layer).dzdx = tanh_ln(res(layer).x,res(layer+1).dzdx );
            otherwise 
                error('net_bp error')    
        end
    end
end
function [ y, dzdw,dzdb,opts] = linear_layer( I,weight,bias,dzdy,opts)
dzdw=[];  
dzdb=[];  
if isempty(dzdy)
    %forward mode
    y=weight*I;
    
    if ~isempty(bias)
        y=y+bias;        
    end    
else    
    %backward mode   
    y=weight'*dzdy;    
    if ~isempty(bias)
        dzdb=mean(dzdy,2);%minibatch averaging    
    end    
    dzdy=permute(dzdy,[1,3,2]);
    I=permute(I,[3,1,2]);
    dzdw=dzdy.*I;
    dzdw=mean(dzdw,3);    
end
end

