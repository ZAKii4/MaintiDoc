# Résultats observés

Produits par `python scripts/evaluate.py`. Ce sont des mesures sur 12 questions, pas une garantie de qualité.

## Mode agent, jeu de réglage : 7/7 contrôles réussis

### 1. Combien de têtes d'attention le modèle de base utilise-t-il, et quelle est la dimension d_k ?

- Contrôle automatique : réussi — 205 s
- Outils : rechercher_tableaux
- Citations : S3 = p.5 (text)

> Le modèle de base utilise h = 8 têtes d'attention [S3]. La dimension d_k est de 64 [S3].

### 2. Quel score BLEU le Transformer (big) obtient-il en anglais-allemand et en anglais-français ?

- Contrôle automatique : réussi — 24 s
- Outils : rechercher_tableaux
- Citations : S1 = p.8 (table)

> Le Transformer (big) obtient un score BLEU de 28.4 pour l'anglais-allemand et un score BLEU de 41.0 pour l'anglais-français [S1].

### 3. Dans le tableau des variantes, quel BLEU obtient-on avec une seule tête d'attention ?

- Contrôle automatique : réussi — 19 s
- Outils : rechercher_tableaux
- Citations : S1 = p.9 (table)

> Le BLEU obtenu avec une seule tête d'attention (h = 1) est de 24.9 [S1].

### 4. Quelle est la complexité par couche de la self-attention comparée à une couche récurrente ?

- Contrôle automatique : réussi — 18 s
- Outils : rechercher_texte
- Citations : S5 = p.6 (table)

> La complexité par couche de la self-attention est de O(n² · d), tandis que celle d'une couche récurrente est de O(n · d²) [S5].

### 5. Montre-moi le schéma de l'architecture du Transformer et décris ses deux parties principales.

- Contrôle automatique : réussi — 37 s
- Outils : rechercher_figures
- Citations : S1 = p.3 (image), S3 = p.3 (text)

> Le schéma de l'architecture du Transformer montre deux parties principales :
> 
> 1. **Encoder** : C'est la première partie de l'architecture. Elle est composée de plusieurs couches identiques, chacune contenant deux sous-couches : une attention multi-têtes et une couche Feed Forward. L'encodage des entrées commence par un encodage positionnel, suivi d'une couche d'embedding. Les sorties de ces couches sont ensuite traitées par des opérations "Add & Norm" et "Feed Forward" [S1][S3].
> 
> 2. **Décodage** : C'est la deuxième partie de l'architecture. Elle est également composée de plusieurs couches identiques, chacune contenant trois sous-couches : une attention multi-têtes sur les sorties de l'encodeur, une attention multi-têtes sur les sorties de la couche précédente, et une couche Feed Forward. Le décodage des sorties commence par un décalage à droite, suivi d'une couche d'embedding. Les sorties de ces couches sont ensuite traitées par des opérations "Add & Norm" et "Softmax" [S1][S3].

### 6. Sur quel matériel et pendant combien de temps les modèles de base ont-ils été entraînés ?

- Contrôle automatique : réussi — 22 s
- Outils : rechercher_texte
- Citations : S1 = p.7 (text)

> Les modèles de base ont été entraînés sur un matériel comprenant un seul ordinateur avec 8 NVIDIA P100 GPUs. Leur entraînement a duré un total de 100 000 étapes ou 12 heures [S1].

### 7. Quelle est la procédure de remplacement du roulement d'une pompe centrifuge ?

- Contrôle automatique : réussi — 23 s
- Outils : rechercher_texte
- Citations : aucune

> Les documents indexés ne permettent pas de répondre à cette question.

## Mode agent, jeu de contrôle : 5/5 contrôles réussis

### 1. Quel optimiseur est utilisé et combien d'étapes de warmup sont prévues ?

- Contrôle automatique : réussi — 24 s
- Outils : rechercher_texte
- Citations : S1 = p.7 (text)

> L'optimiseur utilisé est l'Adam [S1]. Le nombre d'étapes de warmup prévues est de 4000 [S1].

### 2. Quelle valeur de label smoothing est employée pendant l'entraînement ?

- Contrôle automatique : réussi — 23 s
- Outils : rechercher_texte
- Citations : S1 = p.8 (text)

> La valeur de label smoothing employée pendant l'entraînement est ϵls = 0.1 [S1].

### 3. Combien de paramètres, en millions, compte le modèle « big » ?

- Contrôle automatique : réussi — 22 s
- Outils : rechercher_tableaux
- Citations : S1 = p.9 (table)

> Le modèle « big » compte 213 millions de paramètres [S1].

### 4. Quelle est la dimension de la couche interne du réseau feed-forward du modèle de base ?

- Contrôle automatique : réussi — 20 s
- Outils : rechercher_tableaux
- Citations : S1 = p.9 (table)

> La dimension de la couche interne du réseau feed-forward du modèle de base est de 2048 [S1].

### 5. Quel couple de serrage faut-il appliquer aux boulons de fixation du moteur ?

- Contrôle automatique : réussi — 21 s
- Outils : rechercher_texte
- Citations : aucune

> Les documents indexés ne permettent pas de répondre à cette question.

## Mode rag, jeu de réglage : 7/7 contrôles réussis

### 1. Combien de têtes d'attention le modèle de base utilise-t-il, et quelle est la dimension d_k ?

- Contrôle automatique : réussi — 37 s
- Outils : récupération fixe
- Citations : S1 = p.5 (text)

> Le modèle de base utilise h = 8 têtes d'attention, et la dimension d_k est de 64 [S1].

### 2. Quel score BLEU le Transformer (big) obtient-il en anglais-allemand et en anglais-français ?

- Contrôle automatique : réussi — 45 s
- Outils : récupération fixe
- Citations : S6 = p.8 (table)

> Le Transformer (big) obtient un score BLEU de 28.4 en anglais-allemand et un score BLEU de 41.0 en anglais-français [S6].

### 3. Dans le tableau des variantes, quel BLEU obtient-on avec une seule tête d'attention ?

- Contrôle automatique : réussi — 37 s
- Outils : récupération fixe
- Citations : S6 = p.9 (table)

> Dans le tableau des variantes, avec une seule tête d'attention, le BLEU obtenu est de 24.9 [S6].

### 4. Quelle est la complexité par couche de la self-attention comparée à une couche récurrente ?

- Contrôle automatique : réussi — 38 s
- Outils : récupération fixe
- Citations : S6 = p.6 (table)

> La complexité par couche de la self-attention est $ O(n^2 \cdot d) $, tandis que celle d'une couche récurrente est $ O(n \cdot d^2) $ [S6].

### 5. Montre-moi le schéma de l'architecture du Transformer et décris ses deux parties principales.

- Contrôle automatique : réussi — 60 s
- Outils : récupération fixe
- Citations : S8 = p.3 (image), S3 = p.2 (text), S1 = p.3 (text)

> Le schéma de l'architecture du Transformer est représenté dans la Figure 1 [S8]. Cette architecture se compose de deux parties principales : l'encodeur et le décodeur.
> 
> 1. **Encodeur** : L'encodeur est composé d'une pile de N = 6 couches identiques. Chaque couche contient deux sous-couches : la première est un mécanisme d'attention multi-têtes, et la seconde est une couche de réseau de neurones pleinement connectés [S3]. Ces sous-couches sont entourées de connexions résiduelles et de normalisation de couche [S1].
> 
> 2. **Décodeur** : Le décodeur est également composé d'une pile de N = 6 couches identiques. En plus des deux sous-couches présentes dans chaque couche de l'encodeur, le décodeur inclut une troisième sous-couche qui effectue une attention multi-têtes sur la sortie de la pile d'encodeur. De plus, la sous-couche d'attention auto-attention dans le décodeur est modifiée pour empêcher les positions de prêter attention aux positions ultérieures [S1].

### 6. Sur quel matériel et pendant combien de temps les modèles de base ont-ils été entraînés ?

- Contrôle automatique : réussi — 46 s
- Outils : récupération fixe
- Citations : S1 = p.7 (text), S4 = p.7 (text)

> Les modèles de base ont été entraînés sur un matériel comprenant 8 NVIDIA P100 GPUs [S1][S4]. L'entraînement a duré 12 heures [S1].

### 7. Quelle est la procédure de remplacement du roulement d'une pompe centrifuge ?

- Contrôle automatique : réussi — 42 s
- Outils : récupération fixe
- Citations : aucune

> Les documents indexés ne permettent pas de répondre à cette question.

## Mode rag, jeu de contrôle : 5/5 contrôles réussis

### 1. Quel optimiseur est utilisé et combien d'étapes de warmup sont prévues ?

- Contrôle automatique : réussi — 45 s
- Outils : récupération fixe
- Citations : S1 = p.7 (text)

> L'optimiseur utilisé est l'optimiseur Adam [17] avec β1 = 0.9, β2 = 0.98 et ϵ = 10−9. Le nombre d'étapes de warmup prévues est de 4000 [S1].

### 2. Quelle valeur de label smoothing est employée pendant l'entraînement ?

- Contrôle automatique : réussi — 44 s
- Outils : récupération fixe
- Citations : S1 = p.8 (text)

> La valeur de label smoothing employée pendant l'entraînement est ϵls = 0.1 [S1].

### 3. Combien de paramètres, en millions, compte le modèle « big » ?

- Contrôle automatique : réussi — 34 s
- Outils : récupération fixe
- Citations : S6 = p.9 (table)

> Le modèle « big » compte 213 millions de paramètres [S6].

### 4. Quelle est la dimension de la couche interne du réseau feed-forward du modèle de base ?

- Contrôle automatique : réussi — 48 s
- Outils : récupération fixe
- Citations : S2 = p.5 (text)

> La dimension de la couche interne du réseau feed-forward du modèle de base est dff = 2048 [S2].

### 5. Quel couple de serrage faut-il appliquer aux boulons de fixation du moteur ?

- Contrôle automatique : réussi — 44 s
- Outils : récupération fixe
- Citations : aucune

> Les documents indexés ne permettent pas de répondre à cette question.
