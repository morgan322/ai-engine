#include "mainwindow.h"
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QDebug>
#include <QJsonDocument>
#include <QFileDialog>
#include <QGroupBox>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    // 左侧的系统配置
    QLabel* labelSystemConfig = new QLabel("System Configuration");
    QLabel* labelTimeZone = new QLabel("Time Zone");
    m_timeZoneCombo = new QComboBox(this);
    m_timeZoneCombo->addItem("UTC+8");

    QHBoxLayout* layoutTimeZone = new QHBoxLayout;
    layoutTimeZone->addWidget(labelTimeZone);
    layoutTimeZone->addWidget(m_timeZoneCombo);

    QPushButton* buttonSaveParameter = new QPushButton("Save Parameter", this);
    connect(buttonSaveParameter, &QPushButton::clicked, this, &MainWindow::saveParameter);

    m_systemLayout = new QVBoxLayout;
    m_systemLayout->addWidget(labelSystemConfig);
    m_systemLayout->addLayout(layoutTimeZone);
    m_systemLayout->addWidget(buttonSaveParameter);
    m_systemLayout->setStretch(0, 1); // 修改 stretch 因子
    m_systemLayout->setStretch(1, 1);
    m_systemLayout->setStretch(2, 2);

    // 右侧的算法配置
    QLabel* labelAlgorithmConfig = new QLabel("Algorithm Configuration");
    QLabel* labelDataSource = new QLabel("Data Source");
    m_dataSourceCombo = new QComboBox(this);
    m_dataSourceCombo->addItem("Camera");
    m_dataSourceCombo->addItem("Video");

    QHBoxLayout* layoutDataSource = new QHBoxLayout;
    layoutDataSource->addWidget(labelDataSource);
    layoutDataSource->addWidget(m_dataSourceCombo);

    QLabel* labelDetectionBox = new QLabel("Detection Box");
    m_detectionBoxCombo = new QComboBox(this);
    m_detectionBoxCombo->addItem("Square");
    m_detectionBoxCombo->addItem("Circle");

    QHBoxLayout* layoutDetectionBox = new QHBoxLayout;
    layoutDetectionBox->addWidget(labelDetectionBox);
    layoutDetectionBox->addWidget(m_detectionBoxCombo);

    m_algorithmLayout = new QVBoxLayout;
    m_algorithmLayout->addWidget(labelAlgorithmConfig);
    m_algorithmLayout->addLayout(layoutDataSource);
    m_algorithmLayout->addLayout(layoutDetectionBox);
    m_algorithmLayout->setStretch(0, 1); // 修改 stretch 因子
    m_algorithmLayout->setStretch(1, 1);
    m_algorithmLayout->setStretch(2, 2);

    // 底部的日志区域
    QLabel* labelLog = new QLabel("Log");
    QLabel* labelLogLevel = new QLabel("Log Level");
    m_logLevelCombo = new QComboBox(this);
    m_logLevelCombo->addItem("INFO");
    m_logLevelCombo->addItem("WARNING");
    m_logLevelCombo->addItem("ERROR");

    QHBoxLayout* layoutLogLevel = new QHBoxLayout;
    layoutLogLevel->addWidget(labelLogLevel);
    layoutLogLevel->addWidget(m_logLevelCombo);

    QPushButton* buttonPrintLogAndResult = new QPushButton("Print Log and Result", this);
    connect(buttonPrintLogAndResult, &QPushButton::clicked, this, &MainWindow::printLogAndResult);

    m_logEdit = new QPlainTextEdit(this);

    m_resultEdit = new QPlainTextEdit(this);

    m_logLayout = new QVBoxLayout;
    m_logLayout->addWidget(labelLog);
    m_logLayout->addLayout(layoutLogLevel);
    m_logLayout->addWidget(m_logEdit);
    m_logLayout->addWidget(buttonPrintLogAndResult);
    m_logLayout->addWidget(m_resultEdit);
    m_logLayout->setStretch(0, 1); // 修改 stretch 因子
    m_logLayout->setStretch(1, 1);
    m_logLayout->setStretch(2, 3);
    m_logLayout->setStretch(3, 1);
    m_logLayout->setStretch(4, 3);

    // 布局并添加到主窗口
    m_mainLayout = new QVBoxLayout;
    QHBoxLayout* layout = new QHBoxLayout;
    QGroupBox* groupBoxSystemConfig = new QGroupBox(tr("System Configuration"));
    groupBoxSystemConfig->setLayout(m_systemLayout);
    QGroupBox* groupBoxAlgorithmConfig = new QGroupBox(tr("Algorithm Configuration"));
    groupBoxAlgorithmConfig->setLayout(m_algorithmLayout);
    layout->addWidget(groupBoxSystemConfig);
    layout->addWidget(groupBoxAlgorithmConfig);
    layout->addLayout(m_logLayout);
    m_mainLayout->addLayout(layout);

    QWidget* mainWidget = new QWidget(this);
    mainWidget->setLayout(m_mainLayout);
    setCentralWidget(mainWidget);
}

void MainWindow::saveParameter()
{
    saveConfigToJson();
}

void MainWindow::saveConfigToJson()
{
    m_configData["Time Zone"] = m_timeZoneCombo->currentText();
    m_configData["Data Source"] = m_dataSourceCombo->currentText();
    m_configData["Detection Box"] = m_detectionBoxCombo->currentText();
    m_configData["Log Level"] = m_logLevelCombo->currentText();

    QJsonDocument configDocument(m_configData);

    // 选择保存路径
    QString fileName = QFileDialog::getSaveFileName(this,
        tr("Save Config as JSON"), "",
        tr("JSON Files (*.json);;All Files (*)"));
    if (fileName.isEmpty()) {
        return;
    }

    // 保存文件
    QFile jsonFile(fileName);
    if (jsonFile.open(QIODevice::WriteOnly)) {
        jsonFile.write(configDocument.toJson());
        jsonFile.close();
    }
}

void MainWindow::printLogAndResult()
{
    QString logOutput = m_logEdit->toPlainText();
    QString resultOutput = "Algorithm Result:\n"; // 这里可以将算法结果写入 resultOutput
    m_resultEdit->setPlainText(logOutput + "\n\n" + resultOutput);
}
