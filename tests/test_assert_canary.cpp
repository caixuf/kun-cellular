#include <iostream>
#include <cassert>

int main() {
    std::cout << "[CANARY] 正在检验 CI 断言激活状态 (Assertion Liveness Check)...\n";
    // 门禁金丝雀：如果断言在编译器层面处于激活状态，assert(false) 必须触发 SIGABRT (非零退出)
    // CTest 配合 WILL_FAIL TRUE 规则，只有在非零退出时才判定测试通过。
    // 只要有任何环境错误定义 -DNDEBUG 导致断言被剥除，此函数将返回 0，触发 CTest 报警！
    assert(false && "Assertion Canary: Assertions are ACTIVE!");
    return 0;
}
