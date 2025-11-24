# Implementação de Classificadores e Regressores - Alzheimer

## ✅ Fontes de Dados Separadas

### Classificação (SVM + DenseNet Classificação)
**CSV**: `oasis_longitudinal_demographic.csv` (dados demográficos originais)
- Usado por: SVM, DenseNet Classificação
- Features: Textura (GLCM) + Clínicas OASIS (EDUC, MMSE, eTIV, nWBV, ASF, CDR, Visit)
- Dimensão: ~80 features

### Regressão (XGBoost + DenseNet Regressão)
**CSV**: `planilha.csv` (dados com features morfológicas pré-calculadas)
- Usado por: XGBoost, DenseNet Regressão
- Features: Textura (GLCM) + **Morfológicas** (area, perimeter, circularity, eccentricity, solidity, extent, mean_intensity) + Clínicas OASIS
- Dimensão: ~87 features

## 📋 Fluxo de Trabalho Completo

### 1. Preparar Dados
```
Menu: Dados → Preparar Dados (80/20)
```
- Carrega **ambos** CSVs (oasis + planilha)
- Split 80/20 por paciente (estratificado)
- Validação: 20% do treino
- Armazena planilha em `split_info['planilha_df']`

### 2a. Para CLASSIFICAÇÃO (SVM)
```
Dados → Extrair Características (Classificação)
SVM → Treinar SVM
SVM → Avaliar SVM
SVM → Matriz de Confusão
```
**Features**: Textura + Clínicas OASIS (SEM morfológicas)

### 2b. Para REGRESSÃO (XGBoost)
```
Dados → Extrair Características (Regressão)
XGBoost → Treinar XGBoost
XGBoost → Avaliar XGBoost
XGBoost → Análise Temporal
```
**Features**: Textura + **Morfológicas PLANILHA** + Clínicas OASIS

### 3. Para CLASSIFICAÇÃO Profunda (DenseNet)
```
(Não precisa extrair características)
DenseNet → Treinar Classificação
DenseNet → Avaliar Classificação
DenseNet → Curvas Classificação
```
**Entrada**: Imagens RGB 224x224 (usa dados do OASIS para labels)

### 4. Para REGRESSÃO Profunda (DenseNet)
```
(Não precisa extrair características)
DenseNet → Treinar Regressão
DenseNet → Avaliar Regressão
```
**Entrada**: Imagens RGB 224x224 (usa idades do OASIS)

#### 2.1 Regressor Raso (XGBoost)
**Entrada**: Características calculadas (features)
- **Textura**: GLCM multi-distâncias (1,3,5) + múltiplos ângulos + estatísticas + quadrantes (72 features)
- **Morfológicas**: area, perimeter, circularity, eccentricity, solidity, extent, mean_intensity (7 features da planilha.csv)
- **Clínicas**: EDUC, MMSE, eTIV, nWBV, ASF, Visit, Years_since_first, CDR (8 features)
- **Total**: ~87 features combinadas

**Treinamento** (linha 815):
- XGBoost Regressor com 400 árvores
- Early stopping (30 rounds)
- Hiperparâmetros: max_depth=6, lr=0.05, subsample=0.8, regularização L1/L2

**Avaliação** (linha 862):
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)  
- R² Score
- Percentuais de predições dentro de ±5 e ±10 anos
- Gráficos: dispersão (real vs predita) + histograma de erros

**Menu**: XGBoost → Treinar XGBoost / Avaliar XGBoost

#### 2.2 Regressor Profundo (DenseNet121)
**Entrada**: Imagens originais (224x224x3 RGB)
- Preprocessamento: normalização [0,1] + resize + conversão grayscale→RGB

**Arquitetura** (linha 1035):
- Base: DenseNet121 pré-treinada ImageNet (congelada)
- Cabeça: GlobalAveragePooling2D → Dense(128, relu) → Dropout(0.3) → Dense(1, linear)
- Loss: MSE, Métrica: MAE
- Otimizador: Adam (lr=1e-3)
- 5 épocas

**Avaliação** (linha 1125):
- MAE, RMSE, R²
- Gráfico de dispersão idade real vs predita
- Curvas de treino/validação (MAE e MSE)

**Menu**: DenseNet → Treinar Regressão / Avaliar Regressão

### 3. Análise Temporal
**Função**: `analise_temporal()` (linha 896)

**Verifica**:
- Pacientes com múltiplas visitas
- Consistência temporal: idades preditas crescem monotonicamente com as visitas?
- Tolerância: Aceita pequenas oscilações negativas (≥-0.5 anos)

**Saída**:
- Percentual de pacientes com crescimento consistente
- Gráfico de trajetórias (idade predita por visita) para até 12 pacientes

**Menu**: XGBoost → Análise Temporal

### 4. Suficiência das Entradas

#### Features Calculadas (Método Raso)
**Textura (GLCM)**:
- ✅ Captura padrões de textura cerebral
- ✅ Múltiplas distâncias e ângulos aumentam robustez
- ✅ Estatísticas globais e por quadrantes

**Morfológicas (da planilha)**:
- ✅ Área, perímetro, circularidade → forma do cérebro
- ✅ Excentricidade, solidez → compacidade
- ✅ Mean intensity → atrofia/densidade

**Clínicas**:
- ✅ MMSE, CDR → estado cognitivo
- ✅ eTIV, nWBV, ASF → volumetria cerebral
- ✅ EDUC, Visit, Years_since_first → contexto temporal

**Conclusão**: **SIM**, features são suficientes pois combinam informação de textura, forma, volume e contexto clínico.

#### Imagens (Método Profundo)
- ✅ CNN aprende features de alto nível automaticamente
- ✅ Transfer learning (ImageNet) acelera convergência
- ✅ Imagens capturam atrofia, ventrículos, padrões estruturais
- ⚠️ Dataset pequeno (~300 exames) pode limitar generalização

**Conclusão**: **SIM**, mas performance depende de regularização (dropout, early stopping) devido ao tamanho do dataset.

### 5. Análise Temporal - Visitas Posteriores
**Expectativa**: Idades preditas em visitas posteriores ≥ visitas anteriores

**Implementação**:
- Agrupa exames por `Subject ID`
- Ordena por `Visit`
- Verifica se diff(PredictedAge) ≥ -0.5 (tolerância para ruído)
- Calcula percentual de pacientes consistentes

**Visualização**:
- Gráfico de linhas: idade predita vs número da visita
- Uma linha por paciente

**Resultados Esperados**:
- Alto percentual (>70%) indica que o modelo captura progressão temporal
- Baixo percentual sugere overfitting ou features inadequadas

## 📋 Como Usar

### Passo a Passo Completo

1. **Preparar Dados**
   - Menu: Dados → Preparar Dados (80/20)
   - Aguardar carregamento de imagens

2. **Extrair Características**
   - Menu: Dados → Extrair Características
   - Resultado: ~87 features (textura + morfologia + clínica)

3. **Regressor Raso (XGBoost)**
   ```
   XGBoost → Treinar XGBoost
   XGBoost → Avaliar XGBoost  (MAE, RMSE, R², gráficos)
   XGBoost → Análise Temporal (consistência visitas)
   ```

4. **Regressor Profundo (DenseNet)**
   ```
   DenseNet → Treinar Regressão  (5 épocas, curvas automáticas)
   DenseNet → Avaliar Regressão  (MAE, RMSE, R², dispersão)
   ```

## 🔍 Diferenças vs Versão Anterior

### Mudanças Principais
1. **CSV**: `oasis_longitudinal_demographic.csv` → `planilha.csv` (sem separador `;`)
2. **Features Morfológicas**: Novas 7 features da planilha (area, perimeter, etc.)
3. **Dimensão Features**: 80 → ~87 (aumentou robustez)

### Compatibilidade
- ✅ Código anterior de classificação permanece funcional
- ✅ Split de dados mantém mesma lógica (80/20 por paciente)
- ✅ Todas as funções de avaliação (acurácia, sensibilidade, etc.) intactas

## 📊 Métricas de Sucesso

### Regressor Raso (XGBoost)
- **Bom**: MAE < 5 anos, R² > 0.7
- **Aceitável**: MAE < 8 anos, R² > 0.5
- **Temporal**: >70% pacientes com crescimento consistente

### Regressor Profundo (DenseNet)
- **Bom**: MAE < 6 anos, R² > 0.6
- **Aceitável**: MAE < 9 anos, R² > 0.4
- **Nota**: Pode ser inferior ao XGBoost devido ao tamanho do dataset

## 🐛 Troubleshooting

### Erro ao preparar dados
- Verificar se `planilha.csv` está na pasta raiz
- Verificar encoding (UTF-8)

### Features com NaN
- Função `extract_morphological_features` usa defaults seguros
- Valores faltantes são substituídos por medianas

### Análise temporal sem pacientes
- Dataset de teste pode não ter pacientes com múltiplas visitas
- Normal se test_size muito pequeno
