clear all;
cart=cart_pole;
net=net_init_pole(); % initialize the newral net for detail see net_init_pole()function
GAMMA   = 0.99;      % Discount factor for reward calculation 
global EPSILON ;
EPSILON = 0.01;      % for exploration of qlearning
ACTIONS = 2;
MAX_FAILURES=200;      
MAX_STEPS = 200;
failures=0;
success=0;
opts.use_gpu=0;
opts.parameters.mom =0.9;
opts.parameters.lr =1e-1;
opts.parameters.weightDecay=1e-3;
opts.parameters.clip=1e-1;
for increment=1:1000   
    % initialize the input and dzdy for gradient and state is initialize
    % fron cartpole calss which is also random 
    Input=zeros(4,MAX_STEPS);
    opts.dzdy =zeros(ACTIONS,MAX_STEPS);
    state=cart.re_set();
    samples=1;
    res(1).x=state';
    [ net,res,opts] = net_ff(net,res,opts);
    Q_new=res(end).x;
    [V_new,a_new]=max(Q_new);
    Input(:,samples)=state;
    valid=0;
    while ~(valid)
        Q_old=Q_new;
        if rand(1)<EPSILON    
            %Choose action randomly. 
            a_old=randi(ACTIONS);
        else
            %select the highest scored action
            a_old=a_new;
        end
        [state,r,valid]=cart.forward(a_old,increment) ; 
        % here valid mean game's continuity if game is over valid is 1
        if valid==1	
            failed = 1;
            V_new = 0.;            
        else
            failed = 0;      
            res(1).x=state';
            [net,res,opts] = net_ff(net,res,opts);
            Q_new=res(end).x;
            [V_new,a_new]=max(Q_new);
            samples=samples+1;
            Input(:,samples)=state;
            
        end
        % calculation of gradient for last layer
        der=Q_old(a_old)-(r + GAMMA * V_new);
        opts.dzdy(a_old,samples-1)=der;
        % if episode is over and game is failed in between the steps we
        % backpropage and it is batch back propagation
        if failed
            opts.dzdy=opts.dzdy(:,1:samples-1);
            res(1).x=Input(:,samples-1);
            [net,res,opts] = net_ff(net,res,opts);  
            [ net,res,opts ] = net_bp( net,res,opts );    
            [ net,res,opts ] = adam( net,res,opts );
        end
        if success && samples>200
            break;
        end
        if samples>=MAX_STEPS
            success=1;       
       end
    end 
    % to display the running times and total reward for the episode
    disp(['Trial is ' int2str(increment) ' reward is '  num2str(samples-1)]);

end
%% it is mainly for the network making purpose
function net = net_init_pole()
rng('default');
f=1/1 ;
net.layers = {} ;
% 2-layer net
net.layers{end+1} = struct('type', 'mlp', ...
                           'weights', {{f*rand(16,4, 'single'), zeros(16,1,'single')}}) ;
net.layers{end+1} = struct('type', 'relu') ;

net.layers{end+1} = struct('type', 'mlp', ...
                           'weights', {{f*rand(2,16, 'single'), zeros(2,1,'single')}}) ;
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

