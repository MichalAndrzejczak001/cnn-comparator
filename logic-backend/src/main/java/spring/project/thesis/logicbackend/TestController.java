package spring.project.thesis.logicbackend;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api")
public class TestController {
    @GetMapping("/get1")
    public String get1() {
        return "get1";
    }

    @GetMapping("/get2")
    public String get2() {
        return "get2";
    }

    @GetMapping("/get3")
    public String get3() {
        return "get3";
    }

    @GetMapping("/get4")
    public String get4() {
        return "get4";
    }

    @GetMapping("/get5")
    public String get5() {
        return "get5";
    }

    @GetMapping("/get6")
    public String get16() {
        return "get6";
    }

    @GetMapping("/get7")
    public String get17() {
        return "get7";
    }
}
