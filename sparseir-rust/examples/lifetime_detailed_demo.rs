fn main() {
    println!("=== ローカル変数のスライスを返す場合の詳細解説 ===\n");
    
    // 1. 基本的なエラー例
    println!("1. 基本的なエラー例");
    // let bad_slice = bad_function();  // ❌ コンパイルエラー！
    println!("bad_function()はコンパイルエラーになります");
    
    // 2. なぜエラーになるのかの詳細説明
    println!("\n2. なぜエラーになるのかの詳細説明");
    explain_lifetime_error();
    
    // 3. 正しい方法
    println!("\n3. 正しい方法");
    let good_slice = good_function();
    println!("good_function()の結果: {:?}", good_slice);
    
    // 4. ライフタイムの可視化
    println!("\n4. ライフタイムの可視化");
    demonstrate_lifetime();
    
    // 5. 構造体のメンバーを返す場合（安全）
    println!("\n5. 構造体のメンバーを返す場合（安全）");
    let vector = create_sample_vector();
    let norms = vector.get_norms();
    println!("構造体のメンバー: {:?}", norms);
    
    // 6. 静的データを返す場合（安全）
    println!("\n6. 静的データを返す場合（安全）");
    let static_slice = get_static_data();
    println!("静的データ: {:?}", static_slice);
    
    // 7. 引数として受け取ったデータを返す場合（安全）
    println!("\n7. 引数として受け取ったデータを返す場合（安全）");
    let input_vec = vec![10.0, 20.0, 30.0];
    let returned_slice = return_input_slice(&input_vec);
    println!("引数のデータ: {:?}", returned_slice);
    println!("元のデータ: {:?}", input_vec);
    
    // 8. 複雑な例：構造体のメンバーを返す
    println!("\n8. 複雑な例：構造体のメンバーを返す");
    let complex_vector = create_complex_vector();
    let complex_norms = complex_vector.get_norms();
    println!("複雑な構造体のメンバー: {:?}", complex_norms);
    
    // 9. エラーメッセージの解説
    println!("\n9. エラーメッセージの解説");
    explain_error_messages();
}

// ❌ 悪い例：ローカル変数の参照を返そうとする
// fn bad_function() -> &[f64] {
//     let vec = vec![1.0, 2.0, 3.0];
//     &vec  // ❌ エラー！vecは関数終了時に破棄される
// }

// ✅ 良い例：静的データを返す
fn good_function() -> &'static [f64] {
    &[1.0, 2.0, 3.0]  // 静的データなので安全
}

// ライフタイムエラーの詳細説明
fn explain_lifetime_error() {
    println!("ローカル変数のスライスを返すとエラーになる理由:");
    println!("1. ローカル変数は関数終了時に破棄される");
    println!("2. 破棄されたデータへの参照は無効（ダングリングポインタ）");
    println!("3. Rustの所有権システムがこれを自動的に検出");
    println!("4. コンパイル時にエラーとして報告");
}

// ライフタイムの可視化
fn demonstrate_lifetime() {
    println!("ライフタイムの可視化:");
    
    // ローカルスコープでの参照
    {
        let local_vec = vec![1.0, 2.0, 3.0];
        let local_slice = &local_vec;
        println!("ローカルスコープ内: {:?}", local_slice);
        // local_vecはここで破棄される
    } // local_sliceもここで無効になる
    
    println!("ローカルスコープ終了後: 参照は無効");
}

// 構造体のメンバーを返す場合（安全）
#[derive(Debug)]
struct SampleVector {
    norms: Vec<f64>,
    knots: Vec<f64>,
}

impl SampleVector {
    fn new() -> Self {
        Self {
            norms: vec![0.5, 0.7, 0.9],
            knots: vec![0.0, 1.0, 2.0],
        }
    }
    
    // ✅ 安全：構造体のメンバーへの参照を返す
    fn get_norms(&self) -> &[f64] {
        &self.norms  // 構造体が存在する限り有効
    }
    
    fn get_knots(&self) -> &[f64] {
        &self.knots
    }
}

fn create_sample_vector() -> SampleVector {
    SampleVector::new()
}

// 静的データを返す場合（安全）
fn get_static_data() -> &'static [f64] {
    &[100.0, 200.0, 300.0]  // 静的データなので安全
}

// 引数として受け取ったデータを返す場合（安全）
fn return_input_slice(input: &Vec<f64>) -> &[f64] {
    input  // 引数の参照を返すので安全
}

// 複雑な構造体の例
#[derive(Debug)]
struct ComplexVector {
    data: Vec<f64>,
    metadata: Vec<String>,
}

impl ComplexVector {
    fn new() -> Self {
        Self {
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            metadata: vec!["a".to_string(), "b".to_string()],
        }
    }
    
    // ✅ 安全：構造体のメンバーへの参照を返す
    fn get_norms(&self) -> &[f64] {
        &self.data
    }
    
    fn get_metadata(&self) -> &[String] {
        &self.metadata
    }
}

fn create_complex_vector() -> ComplexVector {
    ComplexVector::new()
}

// エラーメッセージの解説
fn explain_error_messages() {
    println!("Rustのエラーメッセージの例:");
    println!("error[E0515]: cannot return reference to local variable `vec`");
    println!("  --> src/main.rs:15:5");
    println!("   |");
    println!("13 | fn bad_function() -> &[f64] {{");
    println!("14 |     let vec = vec![1.0, 2.0, 3.0];");
    println!("15 |     &vec");
    println!("   |     ^^^^");
    println!("   |");
    println!("   = note: returns a reference to data owned by the current function");
    println!();
    println!("このエラーの意味:");
    println!("1. ローカル変数`vec`の参照を返そうとしている");
    println!("2. `vec`は現在の関数が所有しているデータ");
    println!("3. 関数終了時に`vec`は破棄される");
    println!("4. 破棄されたデータへの参照は無効");
    println!("5. Rustがコンパイル時にこれを検出してエラーを報告");
}
