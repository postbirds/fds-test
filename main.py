"""
FDS (Fraud Detection System) 데모
=================================
목적: "FDS는 서버 개발자가 아닌 AI 전문가가 필요하다"는 것을 시연

Kaggle Credit Card Fraud Detection 데이터셋 사용
- 284,807건의 거래 중 492건(0.17%)만 사기
- 극심한 클래스 불균형 문제
"""

import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    f1_score
)
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


def download_data():
    """Kaggle 데이터 다운로드"""
    print("=" * 60)
    print("1단계: 데이터 다운로드")
    print("=" * 60)
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    print(f"데이터 경로: {path}")
    return path


def load_and_explore(path):
    """데이터 로드 및 탐색 - 문제의 복잡성 보여주기"""
    print("\n" + "=" * 60)
    print("2단계: 데이터 탐색 - 왜 단순하지 않은가?")
    print("=" * 60)
    
    df = pd.read_csv(f"{path}/creditcard.csv")
    
    print(f"\n총 거래 건수: {len(df):,}건")
    print(f"피처 수: {df.shape[1]}개")
    
    # 클래스 불균형 - 핵심 문제점 #1
    fraud_count = df['Class'].sum()
    normal_count = len(df) - fraud_count
    fraud_ratio = fraud_count / len(df) * 100
    
    print(f"\n⚠️  [문제점 #1] 극심한 클래스 불균형:")
    print(f"   - 정상 거래: {normal_count:,}건 ({100-fraud_ratio:.2f}%)")
    print(f"   - 사기 거래: {fraud_count:,}건 ({fraud_ratio:.3f}%)")
    print(f"   → 단순히 '모두 정상'이라고 예측해도 99.83% 정확도!")
    print(f"   → Accuracy는 무의미한 지표")
    
    # 피처 설명 - 핵심 문제점 #2
    print(f"\n⚠️  [문제점 #2] PCA 변환된 피처:")
    print(f"   - V1~V28: 보안상 PCA로 변환된 피처 (원본 의미 불명)")
    print(f"   - Time: 첫 거래 이후 경과 시간(초)")
    print(f"   - Amount: 거래 금액")
    print(f"   → 도메인 지식만으로 규칙 만들기 불가능")
    
    return df


def demonstrate_rule_based_failure(df):
    """규칙 기반 접근의 한계 시연"""
    print("\n" + "=" * 60)
    print("3단계: 규칙 기반 접근의 한계 (서버 개발자 방식)")
    print("=" * 60)
    
    # 사기 거래의 금액 분포 확인
    fraud_df = df[df['Class'] == 1]
    normal_df = df[df['Class'] == 0]
    
    print(f"\n사기 거래 금액 통계:")
    print(f"   - 평균: ${fraud_df['Amount'].mean():.2f}")
    print(f"   - 중앙값: ${fraud_df['Amount'].median():.2f}")
    print(f"   - 최대: ${fraud_df['Amount'].max():.2f}")
    
    print(f"\n정상 거래 금액 통계:")
    print(f"   - 평균: ${normal_df['Amount'].mean():.2f}")
    print(f"   - 중앙값: ${normal_df['Amount'].median():.2f}")
    
    # 단순 규칙 테스트: "금액이 X 이상이면 사기"
    print("\n📌 단순 규칙 테스트: '금액 > 임계값이면 사기로 판단'")
    
    thresholds = [100, 500, 1000, 2000]
    
    for threshold in thresholds:
        predicted_fraud = df['Amount'] > threshold
        actual_fraud = df['Class'] == 1
        
        # True Positive: 실제 사기를 사기로 예측
        tp = ((predicted_fraud) & (actual_fraud)).sum()
        # False Positive: 정상을 사기로 예측
        fp = ((predicted_fraud) & (~actual_fraud)).sum()
        # False Negative: 사기를 정상으로 예측
        fn = ((~predicted_fraud) & (actual_fraud)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"\n   임계값 ${threshold}:")
        print(f"      - 사기 탐지율(Recall): {recall*100:.1f}% (492건 중 {tp}건 탐지)")
        print(f"      - 정밀도(Precision): {precision*100:.2f}% (오탐 {fp:,}건)")
        
    print("\n❌ 결론: 단순 금액 규칙으로는 사기 탐지 불가능")
    print("   - 낮은 임계값 → 오탐(False Positive) 폭증")
    print("   - 높은 임계값 → 대부분의 사기 미탐지")


def demonstrate_ml_complexity(df):
    """ML 접근의 복잡성 시연 - 전문가가 필요한 이유"""
    print("\n" + "=" * 60)
    print("4단계: ML 접근법 - 왜 전문가가 필요한가?")
    print("=" * 60)
    
    # 데이터 준비
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # 스케일링 (Amount, Time)
    scaler = StandardScaler()
    X['Amount_scaled'] = scaler.fit_transform(X[['Amount']])
    X['Time_scaled'] = scaler.fit_transform(X[['Time']])
    X = X.drop(['Amount', 'Time'], axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\n학습 데이터: {len(X_train):,}건")
    print(f"테스트 데이터: {len(X_test):,}건")
    
    # === 전문 지식이 필요한 포인트들 ===
    
    print("\n" + "-" * 50)
    print("⚠️  전문가가 결정해야 할 사항들:")
    print("-" * 50)
    
    print("""
    1. 모델 선택
       - Logistic Regression? Random Forest? XGBoost? 
       - Neural Network? Isolation Forest? AutoEncoder?
       
    2. 불균형 데이터 처리 전략
       - SMOTE (Synthetic Minority Over-sampling)?
       - 언더샘플링? 클래스 가중치 조정?
       - Anomaly Detection 방식?
       
    3. 평가 지표 선택
       - Accuracy (❌ 무의미)
       - Precision vs Recall 트레이드오프
       - F1-Score, AUPRC, ROC-AUC?
       
    4. 임계값(Threshold) 최적화
       - 기본 0.5가 최적인가?
       - 비즈니스 비용 고려한 최적점은?
       
    5. 하이퍼파라미터 튜닝
       - Grid Search? Random Search? Bayesian Optimization?
       - Cross-validation 전략은?
    """)
    
    # 모델 비교 시연
    print("\n" + "-" * 50)
    print("📊 모델 성능 비교 (기본 설정)")
    print("-" * 50)
    
    results = []
    
    # 1. Logistic Regression (기본)
    print("\n[1] Logistic Regression (기본 설정)...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_prob_lr = lr.predict_proba(X_test)[:, 1]
    
    results.append({
        'model': 'Logistic Regression',
        'f1': f1_score(y_test, y_pred_lr),
        'auprc': average_precision_score(y_test, y_prob_lr),
        'roc_auc': roc_auc_score(y_test, y_prob_lr)
    })
    
    # 2. Logistic Regression (클래스 가중치)
    print("[2] Logistic Regression (클래스 가중치 적용)...")
    lr_balanced = LogisticRegression(
        max_iter=1000, 
        random_state=42,
        class_weight='balanced'  # 전문가가 알아야 할 옵션
    )
    lr_balanced.fit(X_train, y_train)
    y_pred_lr_bal = lr_balanced.predict(X_test)
    y_prob_lr_bal = lr_balanced.predict_proba(X_test)[:, 1]
    
    results.append({
        'model': 'LR + Class Weight',
        'f1': f1_score(y_test, y_pred_lr_bal),
        'auprc': average_precision_score(y_test, y_prob_lr_bal),
        'roc_auc': roc_auc_score(y_test, y_prob_lr_bal)
    })
    
    # 3. Random Forest (기본)
    print("[3] Random Forest (기본 설정)...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]
    
    results.append({
        'model': 'Random Forest',
        'f1': f1_score(y_test, y_pred_rf),
        'auprc': average_precision_score(y_test, y_prob_rf),
        'roc_auc': roc_auc_score(y_test, y_prob_rf)
    })
    
    # 4. Random Forest (튜닝)
    print("[4] Random Forest (클래스 가중치 + 튜닝)...")
    rf_tuned = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_tuned.fit(X_train, y_train)
    y_pred_rf_tuned = rf_tuned.predict(X_test)
    y_prob_rf_tuned = rf_tuned.predict_proba(X_test)[:, 1]
    
    results.append({
        'model': 'RF + Tuning',
        'f1': f1_score(y_test, y_pred_rf_tuned),
        'auprc': average_precision_score(y_test, y_prob_rf_tuned),
        'roc_auc': roc_auc_score(y_test, y_prob_rf_tuned)
    })
    
    # 결과 출력
    print("\n" + "=" * 60)
    print("📈 성능 비교 결과")
    print("=" * 60)
    
    results_df = pd.DataFrame(results)
    print(f"\n{'모델':<25} {'F1-Score':<12} {'AUPRC':<12} {'ROC-AUC':<12}")
    print("-" * 60)
    for _, row in results_df.iterrows():
        print(f"{row['model']:<25} {row['f1']:.4f}       {row['auprc']:.4f}       {row['roc_auc']:.4f}")
    
    # Confusion Matrix 시각화
    print("\n📊 시각화 생성 중...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    models = [
        ('Logistic Regression (기본)', y_pred_lr),
        ('LR + Class Weight', y_pred_lr_bal),
        ('Random Forest (기본)', y_pred_rf),
        ('RF + Tuning', y_pred_rf_tuned)
    ]
    
    for ax, (name, y_pred) in zip(axes.flatten(), models):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'{name}')
        ax.set_ylabel('실제')
        ax.set_xlabel('예측')
        ax.set_xticklabels(['정상', '사기'])
        ax.set_yticklabels(['정상', '사기'])
    
    plt.suptitle('모델별 Confusion Matrix 비교\n(같은 데이터, 다른 설정 → 완전히 다른 결과)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('confusion_matrix_comparison.png', dpi=150, bbox_inches='tight')
    print("   → confusion_matrix_comparison.png 저장됨")
    
    # Precision-Recall Curve
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for name, y_prob in [
        ('Logistic Regression', y_prob_lr),
        ('LR + Class Weight', y_prob_lr_bal),
        ('Random Forest', y_prob_rf),
        ('RF + Tuning', y_prob_rf_tuned)
    ]:
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        ax.plot(recall, precision, label=f'{name} (AUPRC={ap:.3f})')
    
    ax.set_xlabel('Recall (사기 탐지율)')
    ax.set_ylabel('Precision (정밀도)')
    ax.set_title('Precision-Recall Curve 비교\n(모델/설정에 따라 성능이 크게 달라짐)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('precision_recall_curve.png', dpi=150, bbox_inches='tight')
    print("   → precision_recall_curve.png 저장됨")
    
    return results_df, y_test, y_prob_rf_tuned


def demonstrate_threshold_impact(y_test, y_prob):
    """임계값 조정의 영향 - 비즈니스 결정의 복잡성"""
    print("\n" + "=" * 60)
    print("5단계: 임계값(Threshold) 조정 - 비즈니스 의사결정")
    print("=" * 60)
    
    print("""
    💡 핵심 질문: "사기일 확률이 몇 % 이상이면 차단할 것인가?"
    
    - 임계값 낮춤 (예: 0.3) → Recall ↑ (더 많은 사기 탐지)
                           → Precision ↓ (정상 거래도 많이 차단)
                           → 고객 불만 증가
    
    - 임계값 높임 (예: 0.7) → Precision ↑ (확실한 것만 차단)
                           → Recall ↓ (많은 사기 놓침)
                           → 금전적 손실 증가
    """)
    
    print("\n임계값별 성능 변화:")
    print(f"{'임계값':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 50)
    
    for threshold in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
        y_pred = (y_prob >= threshold).astype(int)
        
        tp = ((y_pred == 1) & (y_test == 1)).sum()
        fp = ((y_pred == 1) & (y_test == 0)).sum()
        fn = ((y_pred == 0) & (y_test == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{threshold:<10} {precision:.4f}       {recall:.4f}       {f1:.4f}")
    
    print("\n→ 최적 임계값 결정에는 비즈니스 비용 분석이 필요")
    print("   (사기 1건 미탐지 비용 vs 정상 거래 1건 차단 비용)")


def print_conclusion():
    """최종 결론"""
    print("\n" + "=" * 60)
    print("📋 최종 결론: 왜 AI 전문가가 필요한가?")
    print("=" * 60)
    
    print("""
    ┌─────────────────────────────────────────────────────────┐
    │  서버 개발자 영역 (규칙 기반)                              │
    ├─────────────────────────────────────────────────────────┤
    │  • IF 금액 > 100만원 THEN 차단                           │
    │  • IF 해외 거래 AND 심야 THEN 알림                        │
    │  • 블랙리스트 IP 차단                                    │
    │                                                         │
    │  → 단순하지만 적응형 사기에 무력                          │
    │  → False Positive 관리 불가                             │
    └─────────────────────────────────────────────────────────┘
    
    ┌─────────────────────────────────────────────────────────┐
    │  AI 전문가 영역 (ML 기반)                                │
    ├─────────────────────────────────────────────────────────┤
    │  • 모델 아키텍처 선택 (앙상블, 딥러닝, 이상탐지)            │
    │  • 불균형 데이터 처리 (SMOTE, 가중치, 샘플링)              │
    │  • Feature Engineering (도메인 지식 + 통계적 기법)        │
    │  • 하이퍼파라미터 최적화 (Bayesian, Grid Search)          │
    │  • 평가 지표 설계 (비즈니스 비용 반영)                     │
    │  • 모델 모니터링 및 재학습 파이프라인                      │
    │  • 설명 가능한 AI (규제 대응)                             │
    │                                                         │
    │  → 논문 수준의 지식 필요                                  │
    │  → 지속적인 연구 및 실험 필요                             │
    └─────────────────────────────────────────────────────────┘
    
    📚 참고할 만한 논문/자료:
    • "Credit Card Fraud Detection: A Realistic Modeling..." (IEEE)
    • "Learning from Imbalanced Data" - He & Garcia
    • Kaggle 대회 상위 솔루션 분석
    
    ✅ 결론: FDS는 단순 '개발'이 아닌 '연구' 영역
           ML 전문가 또는 데이터 사이언티스트 필요
    """)


def main():
    """메인 실행"""
    print("  FDS(Fraud Detection System) 데모")
    
    # 1. 데이터 다운로드
    path = download_data()
    
    # 2. 데이터 탐색
    df = load_and_explore(path)
    
    # 3. 규칙 기반 한계 시연
    demonstrate_rule_based_failure(df)
    
    # 4. ML 복잡성 시연
    results, y_test, y_prob = demonstrate_ml_complexity(df)
    
    # 5. 임계값 영향 시연
    demonstrate_threshold_impact(y_test, y_prob)
    
    # 6. 결론
    print_conclusion()
    
    print("   생성된 파일:")
    print("   - confusion_matrix_comparison.png")
    print("   - precision_recall_curve.png")


if __name__ == "__main__":
    main()