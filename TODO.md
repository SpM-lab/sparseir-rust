
* まずは、twofloat <-> BigFloat間の精度を担保した変換を確立する。特にexp(1), exp(-1)の精度は出るか？
* xprec-svdの精度を確認 (Hilbert matrixを30桁ぐらいで再現できる？)
* CentroSymmetricKernerlに特化したコードを追加. 設計が複雑になるのでReducedKernel等は廃止. +/-1のみを取るParity型はあるか？

pub trait Kernel: Send + Sync {
    fn compute(&self, x: TwoFloat, y: TwoFloat) -> TwoFloat;
    fn compute_f64(&self, x: f64, y: f64) -> f64;
    fn lambda(&self) -> f64;
    fn conv_radius(&self) -> f64;
}

pub trait CentroSymmetricKernelProperties {
    type SVEHintsType<T>: SVEHints<T> + Clone
    where
        T: Copy + Debug + Send + Sync + CustomNumeric + 'static;
    fn ypower(&self) -> i32;
    fn xrange(&self) -> (f64, f64);
    fn yrange(&self) -> (f64, f64);
    fn weight<S: StatisticsType + 'static>(&self, beta: f64, omega: f64) -> f64;
    fn inv_weight<S: StatisticsType + 'static>(&self, beta: f64, omega: f64) -> f64;
    fn sve_hints<T>(&self, epsilon: f64) -> Self::SVEHintsType<T>
    where
        T: Copy + Debug + Send + Sync + CustomNumeric + 'static;
}


pub trait CentroSymmKernel: Send + Sync {
    fn compute(&self, x: TwoFloat, y: TwoFloat) -> TwoFloat;
    fn compute_f64(&self, x: f64, y: f64) -> f64;
    fn compute_reduced(&self, x: TwoFloat, y: TwoFloat, p: Parity) -> TwoFloat;
    fn compute_reduced_f64(&self, x: f64, y: f64, p: Parity) -> f64;
    fn lambda(&self) -> f64;
    fn conv_radius(&self) -> f64;
}

pub trait CentroSymmetricKernelProperties {
    type SVEHintsType<T>: SVEHints<T> + Clone
    where
        T: Copy + Debug + Send + Sync + CustomNumeric + 'static;
    fn ypower(&self) -> i32;
    fn xmax(&self) -> f64;
    fn ymax(&self) -> f64;
    fn weight<S: StatisticsType + 'static>(&self, beta: f64, omega: f64) -> f64;
    fn inv_weight<S: StatisticsType + 'static>(&self, beta: f64, omega: f64) -> f64;
    fn sve_hints<T>(&self, epsilon: f64) -> Self::SVEHintsType<T>
    where
        T: Copy + Debug + Send + Sync + CustomNumeric + 'static;
}



* SparseIR.jlをフォルダに追加した。以降必要であれば、Juliaスクリプトを生成実行し、デバッグ用の参照データを生成し、デバッグに用いよ。単体テストに数値データを組み込むのはできるだけ避けたい。必要がある場合は相談せよ 
* Rustに多倍精度演算のライブラリはあるか？
* 以降は作業プランである。一つずつ完了せよ。それぞれの完了は私が承認する。
* RegularizedBoseKernel, LogisticKernerlの値の計算法が簡略化されている。compute_uvで検索しろ。桁落ちを考慮したアルゴリズムである。この実装に変更しろ。このアルゴリズムを使い、f64の計算はTwoFloatにdelegateせず、f64で完結せよ。結果をlamba = {10.0, 42.0, 10000.0}で多倍精度によるカーネルの表式の評価結果とf64/twofloatの範囲で一致することを確認せよ。
* symm_segmentsの単体テストを実装 x_segments = {-1.0, 0.0, 1.0}, {-1.0, -0.5, 0.5, 1.0}のときに, reducedされた結果が必ず0を含む想定通りの結果になっているか？
* RegularizedBoseSVEHints<T>とLogisticSVEHints<T>に対するsegment_x, segment_yの実装がC++から簡略化されている。修正後、Julia版の出力と比較しろ
* ReducedKernel<InnerKernel>に対するSVEHintsの実装を完了しろ
* C++にあるRegularizedBoseKernelOdd,  LogisticKernelOddが抜けているようだ。おそらくsign = -1のときに桁落ちを防ぐための実装だね。これをRustに導入するためのデザインを検討。
おそらく、SymmetrizedKernel<LogisticKernel, 1>, SymmetrizedKernel<LogisticKerne, -1>に対するcomputeを別に実装すればよさそう。RegularizedBoseKernelOddも同様だね。
