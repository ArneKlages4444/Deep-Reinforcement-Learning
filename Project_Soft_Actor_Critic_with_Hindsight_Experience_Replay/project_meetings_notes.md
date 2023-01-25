## Notes - Meetings Final Project Deep Reinforcement Learning

## 2022-08-12

    - **Meetings**? -> regelmäßige meetings; basierend auf unserer Initiative

    - **environments**: third party environments (default: envs aus originalem HER paper); alternativ: finde env welches uns interessiert

    - für HER: reacher von MuJoCo

    - **resourcen**:

- - könnten auch vor implementierten SAC nehmen, aber gut wenn selbst implementiert

- (- deep reinforcement learning 285 talk zu HER (?))

- - pieter abbeel talk

- 

- **coding questions**:

- - tfp macht reparametrization automatisch

- - extra noise kann entfernt werden, da pulling von distribution

- - wenn probleme mit daten, statt compile eigenen training, in dem optimizer dort gesetzt wird

- - test-environment: double pendulum, swimmer-hopper von MuJoCo

- - use soft plus for sigma - enforcing action bound in paper

- - research alpha and reward scaling for SAC

- - 0.2 for alpha

- 

Links:

    - https://www.youtube.com/playlist?list=PL_iWQOsE6TfXxKgI1GgyV1B_Xa0DxE5eH

    - https://arxiv.org/pdf/1801.01290.pdf#cite.haarnoja2017reinforcement

## 2022-08-24

#### SAC

- SAC works now -> without value network (update variant)

- MSE discussion (mse * 0.5?) -> stay how it is no noticable difference

- `@tf_funcitons`

- foward pass wrap

-  Gradient tape should be wrapped (don't use `.append` on lists)

- replay buffer can not be wrapped

- `train_step_actor` was reviewed and approved

- train step functions should always have `@tf_function`

- `sample_actions_form_policy` MultivariatNormal or MultivariatNormalDiag??

- ususal normal would need dreiecks matrix 

- reduce sum

#### Environment

- Reacher environment (docs are down)

- Finger exactly on  point vs \epsilon error

- having an area is **necessary**

- Panda gym will be tryed out 

- maybe we can look at the reacher code

- maybe we can build "fork" and change it to our requirements

- better episodic learning then continous learning task

- https://www.reddit.com/r/MachineLearning/comments/oss2e3/n_openai_gym_is_now_actively_maintained_again_by/

- [https://www.gymlibrary.dev/](https://www.gymlibrary.dev/) 

- [GitHub - Farama-Foundation/gym-docs: Code for Gym documentation website](https://github.com/Farama-Foundation/gym-docs)

#### HER

- sample points from trajectory as goal states

- Strategies we implement:

- final strategy -> really only last state

- what would happen if we make decaying rewards over the episode ?

- -> problem scale rewards accordingly

- future strategy -> copy/paste the episode (and clip the end for the replay? -> NO! full cequence is used)

- *IF we have time for experiements* we can add `k` to in final stratege and see how it changes

- we could normalize the rewards to get it between -1 and 1

- originally ~160000 episodes -> not to many :)

### Paper

- We can include math notation again

- 10 pages is usually to much

- implementation part can be extensive

- pseudo code can be extensive

- Graphic need a confidence

- how many runs do we need?

- caption: "We plot this without confidence intervals because every run needs 24h compute time"

- up until 5 runs plot them on theirown

- give area of mean

- recommendation plot few runs on their own, if it there are more runs we can do shaded area (https://seaborn.pydata.org/examples/errorband_lineplots.html)

- we can put the raw data into the repo (as csv or data frame)

## 2022-09-01

[final_project · main · UOS_IKW_DRL_2022_Group37 / homework · GitLab](https://gitlab.gwdg.de/uos_ikw_drl_2022_group37/homework/-/tree/main/final_project)

[Hindsight Experience Replay](https://sites.google.com/site/hindsightexperiencereplay/)

#### Video

- was ist der hintergrund?

- was haben wir genau gemacht

- nicht auf quellcode eingehen

- ...

- selbst überlassen was wir sonst/genau mit dem video machen

- gedacht für leute die vlt noch nicht so viel ahnung vom topic haben

- locker aber wissenschaftlich bleiben

- studentisch wissenschaftlich erklären

- video in readme auf github packen

#### Paper

- genau aufschreiben: was plotten wir genau, was ist der error bar genau? => unter plot schreiben

- in caption am besten beschreiben was abgebildet ist; im text figure referencen und daraus dann schlüsse ziehen

- wie viel code in paper?:

- code wenn es sinn ergibt damit leichter sachen zu beschreiben

- änderungen entweder mit pseudocode oder code beschreiben (Abhängig von länge von pseudo-/quellcode)

- erstellen
   von netzwerken: classisch: darstellung durch graphic, können aber auch 
  durch text/functions darstellen in diesem kontext

- nur relevanten/grundlegenden code beschreiben/erklären -> highlevel idee + relevanten teile

## TODO

Bis zum **08.09.2022 um 12:00 Uhr** **mittags**

- readme

- Video "script" schreibe & Präsi machen -> "two minute paper" - style?

- video

- Paper fertig schreiben

- Paper korrektur
  
  
