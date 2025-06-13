## 계층형 VAE

**계층형 VAE (Hierarchical Variational Autoencoder, Hierarchical VAE)** 는 단일 VAE에 비해 복잡한 데이터의 latent structure를 보다 정밀하게 학습하기 위해 다층적인 latent variable를 도입한 확장된 VAE 모델이다.

일반적인 VAE가 단일 latent space에 기반해 데이터를 재구성하는 반면, 계층형 VAE는 상위 및 하위 수준의 latent variable들을 통해 데이터의 계층적인 특징을 포착하며, 보다 표현력 있는 생성 모델을 구성할 수 있다.

이 글에서는 계층 수가 $T$ 인 계층형 VAE의 구조를 알아보고, 그에 해당하는 *ELBO(Evidence Lower Bound)* 와 loss function을 수학적으로 단계별로 도출해낼 것이다.

### 기호 정의
이전 변수 $z_{i-1}$를 다음 변수 $z_i$로 인코딩하는 분포를 $q_{\phi_i}(z_i|z_{i-1})$라 하고, 반대로 $z_i$를 $z_{i-1}$로 디코딩 하는 분포를 $p_{\theta_i}(z_{i-1}|z_i)$라 하자.

$$
\mathbf{z}=\{z_1,z_2,\cdots,z_T \}\quad\mathbf{\theta}=\{\theta_1,\theta_2,\cdots,\theta_T\}\quad\mathbf{\phi}=\{\phi_1,\phi_2,\cdots,\phi_T\}
$$

별도로, 편의를 위해 $z_0=x$라 가정하자.

## 계층형 VAE ELBO
우선, latent variable이 $z$ 하나인 단일 VAE의 ELBO는 다음과 같이 서술된다.

$$
\text{ELBO}(x;\theta,\phi)=\int q_\phi (z|x)\log\frac{p_\theta (x,z)}{q_\phi(z|x)}\,dz=\mathbb{E}_{q_\phi(z|x)}\left[ \log\frac{p_\theta(x,z)}{q_\phi(z|x)} \right]
$$

이 식을 확장하여, latent variable이 총 $T$개인 $\mathbf{z}=\{z_1,z_2,\cdots,z_T\}$ 계층형 VAE의 ELBO는 다음과 같이 서술할 수 있다.

$$
\text{ELBO}(\mathbf{x};\theta,\phi)=\int_\mathbf{z} q_\phi (\mathbf{z}|\mathbf{x})\log\frac{p_\theta (\mathbf{x},\mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})}\,d\mathbf{z}=\mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}\left[ \log\frac{p_\theta(\mathbf{x},\mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})} \right]
$$

### 마르코프 성질
**마르코프 성질(Markov property)** 이란 어떤 확률 과정에서 현재 상태가 미래를 예측하는데 충분하며, 과거는 영향을 미치지 않는다는 성질을 말한다.

즉, 다음 상태는 오직 현재 상태에만 의존하고, 이전 상태들의 정보는 필요하지 않다. 수식으로는 $P(X_{t+1}|X_t,X_{t-1},\cdots,X_0)=P(X_{t+1}|X_t)$로 표현할 수 있다. 이 성질을 만족하는 확률 과정을 마르코프 과정이라고 한다.

이 마르코프 성질은 계층형 VAE에 적용할 수 있다.

$$
p_\theta(\mathbf{x},\mathbf{z})=p(z_T)\prod_{i=1}^T p_{\theta_i}(z_{i-1}|z_i)
$$

마찬가지로,

$$
q_\phi(\mathbf{z}|\mathbf{x})=\prod_{i=1}^T q_{\phi_i}(z_i|z_{i-1})
$$

와 같이 전개하여 간략하게 표현할 수 있다.

## 계층형 VAE ELBO 전개

계층형 VAE의 ELBO는 다음과 같이 전개할 수 있다. (편의상 $\theta$와 $\phi$의 아래첨자는 생략)

$$
\text{ELBO}(\mathbf{x};\theta,\phi)
= \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}\left[\log\frac{p_\theta(\mathbf{x},\mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})} \right]
$$

$$
= \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}\left[\log\frac{p(z_T)\prod_{i=1}^T p_{\theta_i}(z_{i-1}|z_i)}{\prod_{i=1}^T q_{\phi_i}(z_i|z_{i-1})}\right] \quad (\because\text{Markov prop.})
$$

$$
= \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}\left[\log\left(p_\theta(\mathbf{x}|z_1)\cdot\frac{p(z_T)}{q_\phi(z_T|z_{T-1})}\cdots\frac{p_\theta(z_1|z_2)}{q_\phi(z_1|\mathbf{x})} \right)\right]
$$

즉, 아래와 같이 계층형 VAE의 ELBO는 총 3개의 항($J_0$, $J_1$, $J_i$)의 합으로 나타낼 수 있다.

$$
J_0 = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}\left[\log p_\theta(\mathbf{x}|z_1)\right]
$$

$$
J_T = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}\left[\log\frac{p(z_T)}{q_\phi(z_T|z_{T-1})}\right]
$$

$$
J_i = \sum_{i=1}^{T-1}\mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}\left[\log\frac{p_\theta(z_i|z_{i+1})}{q_\phi(z_i|z_{i-1})}\right], \quad \text{for } i \in [1, T-1]
$$

### 1️⃣ $J_0$ 전개

$$
J_0 = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}\left[\log p_\theta(\mathbf{x}|z_1) \right]
$$

$$
= \int_\mathbf{z} q_\phi(\mathbf{z}|\mathbf{x}) \log p_\theta(\mathbf{x}|z_1)\, d\mathbf{z}
$$

$$
= \int_\mathbf{z} \prod_{i=1}^T q_{\phi_i}(z_i|z_{i-1}) \cdot \log p_\theta(\mathbf{x}|z_1)\, d\mathbf{z}
$$

$$
= \int q_{\phi_1}(z_1|\mathbf{x}) \left[ \prod_{i=2}^T \int q_{\phi_i}(z_i|z_{i-1})\, dz_i \right] \log p_\theta(\mathbf{x}|z_1)\, dz_1
$$

에서 $\prod_{i=2}^T\int q_{\phi_i}(z_i|z_{i-1})dz_i$의 값은 $1$이므로,

$$
J_0=\int q_{\phi_1}(z_1|\mathbf{x})\log p_\theta(\mathbf{x}|z_1)dz_1=\mathbb{E}_{q_{\phi_1}(z_1|\mathbf{x})}\left[\log p_{\theta_1}(\mathbf{x}|z_1)\right]
$$

즉, $J_0$은 $q_{\phi_1}(z_1|\mathbf{x})$에 대한 $\log p_{\theta_1}(\mathbf{x}|z_1)$의 기댓값으로 볼 수 있다.

### 2️⃣ $J_T$ 전개

$$
J_T = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}\left[\log\frac{p(z_T)}{q_\phi(z_T|z_{T-1})}\right]
$$

$$
= \int_\mathbf{z} q_\phi(\mathbf{z}|\mathbf{x}) \log\frac{p(z_T)}{q_\phi(z_T|z_{T-1})} \, d\mathbf{z}
$$

$$
= -\int_\mathbf{z} \prod_{i=1}^T q_{\phi_i}(z_i|z_{i-1}) \cdot \log\frac{q_\phi(z_T|z_{T-1})}{p(z_T)} \, d\mathbf{z}
$$

$$
= -\iint q_\phi(z_{T-1}|z_{T-2})\, q_\phi(z_T|z_{T-1}) \log\frac{q_\phi(z_T|z_{T-1})}{p(z_T)} \, dz_T \, dz_{T-1}
\left[ \prod_{i=1}^{T-2} \int q_{\phi_i}(z_i|z_{i-1})\, dz_i \right]
$$

에서 $\prod_{i=1}^{T-2}\int q_{\phi_i}(z_i|z_{i-1})dz_i$의 값은 $1$이므로,

$$
J_T = -\iint q_\phi(z_{T-1}|z_{T-2}) \left[ q_\phi(z_T|z_{T-1}) \log\frac{q_\phi(z_T|z_{T-1})}{p(z_T)} \, dz_T \right] dz_{T-1}
$$

$$
= -\mathbb{E}_{q_\phi(z_{T-1}|z_{T-2})} \left[ D_{KL}\left( q_\phi(z_T|z_{T-1}) \,\|\, p(z_T) \right) \right]
$$

즉, $J_T$는 $q_\phi(z_{T-1}|z_{T-2})$에 대한 $q_\phi(z_T|z_{T-1})$와 $p(z_T)$의 분포적 유사도($D_{KL}$; KL-발산값)의 음의 기댓값으로 볼 수 있다.

### 3️⃣ $J_i$ 전개

$J_i$ 값은 $i\in[1,T-1]$에 따라 $J_1,\cdots,J_{T-1}$ 총 $T-1$개의 하위 항들의 합으로 구성되어 있다.

$$
J_i = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}\left[\log\frac{p_\theta(z_i|z_{i+1})}{q_\phi(z_i|z_{i-1})}\right]
$$

$$
= \int_\mathbf{z} q_\phi(\mathbf{z}|\mathbf{x}) \log\frac{p_\theta(z_i|z_{i+1})}{q_\phi(z_i|z_{i-1})} \, d\mathbf{z}
$$

$$
= -\int_\mathbf{z} \prod_{j=1}^T q_{\phi_j}(z_j|z_{j-1}) \cdot \log\frac{q_\phi(z_i|z_{i-1})}{p_\theta(z_i|z_{i+1})} \, d\mathbf{z}
$$

$$
= -\iint q_\phi(z_{i+1}|z_i)\, q_\phi(z_i|z_{i-1}) \log\frac{q_\phi(z_i|z_{i-1})}{p_\theta(z_i|z_{i+1})} \, dz_i \, dz_{i+1}
\left[ \prod_{j \in \mathbf{z} \setminus \{z_i, z_{i+1}\}} \int q_{\phi_j}(z_j|z_{j-1}) \, dz_j \right]
$$

에서 $\prod_{j\in\mathbf{z}\setminus\left\{z_i,z_{i+1}\right\}}\int q_{\phi_j}(z_j|z_{j-1})dz_j$의 값은 $1$이므로, 

$$
J_i = -\iint q_\phi(z_{i+1}|z_i) \left[ q_\phi(z_i|z_{i-1}) \log\frac{q_\phi(z_i|z_{i-1})}{p_\theta(z_i|z_{i+1})} \, dz_i \right] dz_{i+1}
$$

$$
= -\mathbb{E}_{q_\phi(z_{i+1}|z_i)} \left[ D_{KL}\left( q_\phi(z_i|z_{i-1}) \,\|\, p_\theta(z_i|z_{i+1}) \right) \right]
$$

$$
\forall i, \quad i \in [1, T-1]
$$

즉, $J_i$는 $q_\phi(z_{i+1}|z_i)$에 대한 $q_\phi(z_i|z_{i-1})$와 $p_\theta(z_i|z_{i+1})$의 분포적 유사도의 음의 기댓값으로 볼 수 있다.

### ✅ 최종 ELBO 식

따라서, $T$개의 latent variable을 가진 계층형 VAE의 ELBO는 다음과 같다.

$$
\text{ELBO}(\mathbf{x};\theta,\phi) =
\mathbb{E}_{q_\phi(z_1|\mathbf{x})} \left[ \log p_\theta(\mathbf{x}|z_1) \right]

- \mathbb{E}_{q_\phi(z_{T-1}|z_{T-2})} \left[ D_{KL}\left( q_\phi(z_T|z_{T-1}) \,\|\, p(z_T) \right) \right]

- \sum_{i=1}^{T-1} \mathbb{E}_{q_\phi(z_{i+1}|z_i)} \left[ D_{KL}\left( q_\phi(z_i|z_{i-1}) \,\|\, p_\theta(z_i|z_{i+1}) \right) \right]
$$



## 몬테카를로 방법을 이용한 ELBO 근사
**몬테카를로 방법(Monte Carlo method)** 은 어떤 확률분포에서의 기댓값을 **무작위 표본의 평균** 으로 근사하는 기법이다. 

예를 들어 $\mathbb{E}\left[f(X)\right]$를 계산할 때, $X\sim p(x)$에서 샘플 $x_1,\cdots,x_n$을 추출해 $\frac{1}{N}\sum_{i=1}^N f(x_i)$로 근사한다. 표본 수가 많아질수록 근사 정확도가 높아지지만, 이 경우에서는 $N=1$만으로도 충분히 정확한 근삿값을 얻어낼 수 있다.

참고로, 각 $z_i$는 **재매개변수화 트릭(reparameterization trick)** 을 이용해 $q_\phi(z_i|z_{i-1})$에서 샘플링한다.

이를 이용해 계층형 VAE의 ELBO를 근사하면

$$
\text{ELBO}(\mathbf{x};\theta,\phi) \approx \log p_\theta(\mathbf{x}|z_1)
- D_{KL}\left( q_\phi(z_T|z_{T-1}) \,\|\, p(z_T) \right)
- \sum_{i=1}^{T-1} D_{KL}\left( q_\phi(z_i|z_{i-1}) \,\|\, p_\theta(z_i|z_{i+1}) \right)
$$

$$
\text{ELBO}(\mathbf{x};\theta,\phi) \approx \tilde{J_0}+\tilde{J_T}+\tilde{J_i}
$$

와 같이 표현할 수 있다.

여기서 $\tilde{J_0}$는 $\mathbf{x}$와 $z_1$사이에서의 **재구성 오차항** 의 역할을, $\tilde{J_T}$는 **사전 분포에 대한 무결성항** 의 역할을, $\tilde{J_i}$은 중간 latent variable들 **사이에서의 무결성항** 의 역할을 한다.

### 1️⃣ $\tilde{J_0}$ 근사
우선 디코더를 통해 $z_1$에서 $\hat{\mathbf{x}}$를 추정하고, 이를 이용해 $p_\theta(\mathbf{x}|z_1)$을 정규분포를 이용해 근사한다.

$$
\hat{\mathbf{x}} = \text{Decoder}(z_1, \theta)
$$

$$
p_\theta(\mathbf{x}|z_1) = \mathcal{N}(\mathbf{x}; \hat{\mathbf{x}}, \mathbf{I})
$$

$$
\therefore \log p_\theta(\mathbf{x}|z_1) = -\frac{1}{2} \sum_{d=1}^D (x_d - \hat{x}_d)^2 + \text{const}
$$

$D$는 $\mathbf{x}$와 $\hat{\mathbf{x}}$의 차원 수이다.

### 2️⃣ $\tilde{J_T}$ 근사
사전 분포 $p(z_T)$는 다음과 같이 인코더로부터 추출한 정규분포의 평균($\mathbf{\mu}$)와 표준편차($\mathbf{\sigma}$)를 통해 구할 수 있다.

$$
q_\phi(z_T|z_{T-1}) = \mathcal{N}(z_T;\mathbf{\mu}_T,\mathbf{\sigma}_T^2\mathbf{I})
$$

$$
p(z_T) = \mathcal{N}(z_T;\mathbf{0},\mathbf{I})
$$

따라서 $q_\phi(z_T|z_{T-1})$와 $p(z_T)$의 KL-발산값은 다음과 같이 구할 수 있다.

$$
\therefore D_{KL}\left(q_\phi(z_T|z_{T-1})~||~p(z_T)\right)=-\frac{1}{2}\sum_{h=1}^H\left(1+\log\sigma_{T,h}^2-\mu_{T,h}^2-\sigma_{T,h}^2\right)
$$

$H$는 $z_i$의 차원 수이다.

#### 💡 정규분포의 KL-발산

$q(x)=\mathcal{N}(z;\mathbf{\mu}_1,\mathbf{\sigma}_1^2\mathbf{I})$이고 $p(x)=\mathcal{N}(z;\mathbf{\mu}_2,\mathbf{\sigma}_2^2\mathbf{I})$일 때,

$$
D_{KL}(q~||~p) = -\frac{1}{2} \sum_{h=1}^H \left(1 + \log\frac{\sigma_{1,h}^2}{\sigma_{2,h}^2} - \frac{(\mu_{1,h} - \mu_{2,h})^2}{\sigma_{2,h}^2} - \frac{\sigma_{1,h}^2}{\sigma_{2,h}^2} \right)
$$

$H$는 $x$의 차원 수


### 3️⃣ $\tilde{J_i}$ 근사

앞선 방법과 유사하게,

$$
q_\phi(z_i|z_{i-1}) = \mathcal{N}(z_i;\mathbf{\mu}_i,\mathbf{\sigma}_i^2\mathbf{I})
$$

$$
\hat{z_i} = \text{Decoder}(z_{i+1}, \theta)
$$

$$
p_\theta(z_i|z_{i+1}) = \mathcal{N}(z_i; \hat{z_i}, \mathbf{I})
$$

$$
\therefore\ D_{KL}\left(q_\phi(z_i|z_{i-1}) \,\|\, p_\theta(z_i|z_{i+1})\right)
= -\frac{1}{2} \sum_{h=1}^H \left(1 + \log \sigma_{i,h}^2 - (\mu_{i,h} - \hat{z}_{i,h})^2 - \sigma_{i,h}^2 \right)
$$

마찬가지로 $H$는 $z_j$의 차원 수이다.

### ✅ 최종 ELBO 근사 식

이를 종합하면, $T$개의 latent variable을 가진 계층형 VAE의 ELBO에 대한 근사 식은 다음과 같다.

$$
\text{ELBO}(\mathbf{x};\theta,\phi) \approx -\frac{1}{2} \sum_{d=1}^D (x_d - \hat{x}_d)^2 + \text{const}+\frac{1}{2} \sum_{h=1}^H \left(1 + \log \sigma_{T,h}^2 - \mu_{T,h}^2 - \sigma_{T,h}^2 \right)+\frac{1}{2} \sum_{i=1}^{T-1} \sum_{h=1}^H \left(1 + \log \sigma_{i,h}^2 - (\mu_{i,h} - \hat{z}_{i,h})^2 - \sigma_{i,h}^2 \right)
$$

## 손실 함수(Loss Function)

근사 ELBO로부터 일반화된 계층형 VAE의 **최종적인 손실 함수** 를 도출해 낼 수 있다.

$$
\mathcal{L}(\mathbf{x}; \theta, \phi) = \sum_{d=1}^D (x_d - \hat{x}_d)^2-\sum_{h=1}^H \left(1 + \log \sigma_{T,h}^2 - \mu_{T,h}^2 - \sigma_{T,h}^2 \right)-\sum_{i=1}^{T-1} \sum_{h=1}^H \left(1 + \log \sigma_{i,h}^2 - (\mu_{i,h} - \hat{z}_{i,h})^2 - \sigma_{i,h}^2 \right)
$$
