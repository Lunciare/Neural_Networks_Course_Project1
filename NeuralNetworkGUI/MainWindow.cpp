#include "MainWindow.h"
#include "ui_mainwindow.h"

#include <QMessageBox>
#include <QTimer>
#include <QKeyEvent>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    auto *layersVal = new QIntValidator(1, 5, this);
    ui->layersLineEdit->setValidator(layersVal);

    auto *epochsVal = new QIntValidator(1, 30, this);
    ui->epochsLineEdit->setValidator(epochsVal);

    ui->activationComboBox->installEventFilter(this);
    ui->optimizerComboBox ->installEventFilter(this);
    setFixedSize(400, 350);

    connect(ui->startButton,        &QPushButton::clicked,           this, &MainWindow::onStartClicked);
    connect(ui->layersLineEdit,     &QLineEdit::editingFinished,     this, &MainWindow::validateInputs);
    connect(ui->epochsLineEdit,     &QLineEdit::editingFinished,     this, &MainWindow::validateInputs);
    connect(ui->activationComboBox, &QComboBox::currentIndexChanged, this, &MainWindow::validateInputs);
    connect(ui->optimizerComboBox,  &QComboBox::currentIndexChanged, this, &MainWindow::validateInputs);

    connect(ui->layersLineEdit, &QLineEdit::textChanged, this, [this]() {
      ui->activationComboBox->setEnabled(ui->layersLineEdit->hasAcceptableInput());
      ui->optimizerComboBox ->setEnabled(ui->layersLineEdit->hasAcceptableInput());
      ui->epochsLineEdit    ->setEnabled(ui->layersLineEdit->hasAcceptableInput());
    });

    connect(ui->epochsLineEdit, &QLineEdit::textChanged, this, [this]() {
      ui->activationComboBox->setEnabled(ui->epochsLineEdit->hasAcceptableInput());
      ui->optimizerComboBox ->setEnabled(ui->epochsLineEdit->hasAcceptableInput());
    });

    connect(ui->layersLineEdit, &QLineEdit::returnPressed, this, [this]() {
        ui->activationComboBox->setFocus();
    });

    connect(ui->epochsLineEdit, &QLineEdit::returnPressed, this, [this]() {
        ui->epochsLineEdit->clearFocus();
        updateStartButtonState();
    });

    updateStartButtonState();
}

MainWindow::~MainWindow()
{
    delete ui;
}

bool MainWindow::validateInputs()
{
    QString text1 = ui->layersLineEdit->text();
    if (text1.isEmpty()) return false;

    bool ok;
    int layers = text1.toInt(&ok);
    if (!ok || layers < 1 || layers > 5) {
        QMessageBox::warning(this, "Error",
                             "Please enter an integer from 1 to 5 for number of hidden layers.");
        return false;
    }

    QString text2 = ui->epochsLineEdit->text();
    if (text2.isEmpty()) return false;

    int epochs = text2.toInt(&ok);
    if (!ok || epochs < 1 || epochs > 30) {
        QMessageBox::warning(this, "Invalid input",
                             "Please enter an integer from 1 to 30 for number of epochs.");
        return false;
    }

    if (ui->activationComboBox->currentIndex() == 0) {
        QMessageBox::warning(this, "Missing input",
                             "Please select at least one activation function.");
        return false;
    }

    if (ui->optimizerComboBox->currentIndex() == 0) {
        QMessageBox::warning(this, "Missing input",
                             "Please select an optimizer.");
        return false;
    }

    updateStartButtonState();

    return true;
}

void MainWindow::updateStartButtonState()
{
    bool enabled = true;
    bool ok1, ok2;
    int layers = ui->layersLineEdit->text().toInt(&ok1);
    int epochs = ui->epochsLineEdit->text().toInt(&ok2);
    if (!ok1 || layers < 1 || layers > 5)            enabled = false;
    if (!ok2 || epochs < 1 || epochs > 30)           enabled = false;
    if (ui->activationComboBox->currentIndex() == 0) enabled = false;
    if (ui->optimizerComboBox->currentIndex() == 0)  enabled = false;

    ui->startButton->setEnabled(enabled);
}

void MainWindow::onStartClicked()
{
    if (!validateInputs()) return;

    ui->trainingStatus->setText("Training started...");
    ui->progressBar->setValue(0);
    QTimer *timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, [=]() mutable {
        int value = ui->progressBar->value();
        if (value >= 100) {
            timer->stop();
            ui->trainingStatus->setText("Training complete.");
            ui->mseOutput->setText("0.023");
        } else {
            ui->progressBar->setValue(value + 10);
        }
    });
    timer->start(300);
}

void MainWindow::on_restartButton_clicked()
{
    ui->layersLineEdit->clear();
    ui->epochsLineEdit->clear();

    ui->activationComboBox->setCurrentIndex(0);
    ui->optimizerComboBox->setCurrentIndex(0);

    ui->trainingStatus->setText("Not started");
    ui->progressBar->setValue(0);
    ui->mseOutput->setText("--");

    ui->startButton->setEnabled(false);
}

bool MainWindow::eventFilter(QObject *obj, QEvent *event)
{
    if (event->type() == QEvent::KeyPress) {
        QKeyEvent *keyEvent = static_cast<QKeyEvent *>(event);
        if (keyEvent->key() == Qt::Key_Return || keyEvent->key() == Qt::Key_Enter) {
            if (obj == ui->activationComboBox) {
                ui->optimizerComboBox->setFocus();
                return true;
            } else if (obj == ui->optimizerComboBox) {
                ui->epochsLineEdit->setFocus();
                return true;
            }
        }
    }
    return QMainWindow::eventFilter(obj, event);
}
